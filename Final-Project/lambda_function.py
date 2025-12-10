import os
import json
import hmac
import hashlib
import base64
import datetime
import urllib.request

# 如果之後 JSON 變大，可以考慮做簡單的記憶體快取
ACTIVITY_JSON_FILE = "taipei_activities.json"
ACTIVITY_CACHE = None  # 模組級快取，Lambda 冷啟後可重複使用

# === 環境變數設定 ===
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


# === LINE 簽名驗證 ===
def verify_signature(channel_secret: str, body: str, signature: str) -> bool:
    """
    驗證來自 LINE Webhook 的 X-Line-Signature
    """
    mac = hmac.new(
        channel_secret.encode("utf-8"),
        body.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expected, signature)


# === 回覆 LINE 訊息 ===
def reply_message(reply_token: str, text: str):
    """
    使用 LINE Reply API 回覆文字訊息
    """
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}",
    }
    body = {
        "replyToken": reply_token,
        "messages": [
            {
                "type": "text",
                "text": text,
            }
        ],
    }
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        print("[DEBUG] calling LINE reply API...")
        # 給個 2 秒 timeout，避免卡太久
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            resp.read()
        print("[DEBUG] LINE reply done")
    except Exception as e:
        print(f"[ERROR] LINE reply failed: {e}")


# === 從本地 JSON 檔載入活動資料 ===
def load_activities_from_file() -> dict:
    """
    從同層目錄的 taipei_activities.json 載入活動資料。
    使用模組級快取避免每次都重新讀檔。
    """
    global ACTIVITY_CACHE

    if ACTIVITY_CACHE is not None:
        return ACTIVITY_CACHE

    try:
        file_path = os.path.join(os.path.dirname(__file__), ACTIVITY_JSON_FILE)
    except NameError:
        # 在某些執行環境 __file__ 不存在時，退而求其次使用目前工作目錄
        file_path = ACTIVITY_JSON_FILE

    try:
        print(f"[DEBUG] loading activities from {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 簡單驗證結構
        if not isinstance(data, dict) or "activities" not in data:
            raise ValueError("Invalid JSON structure: missing 'activities'")
        ACTIVITY_CACHE = data
        print(f"[DEBUG] loaded {len(data.get('activities', []))} activities")
        return data
    except Exception as e:
        print(f"[ERROR] load_activities_from_file failed: {e}")
        # 回傳一個安全的空結構
        return {"last_updated": None, "activities": [], "error": str(e)}


# === 日期工具與時間篩選邏輯 ===
def _parse_date(date_str: str):
    """把 'YYYY-MM-DD' 字串轉成 datetime.date，失敗回 None"""
    if not date_str:
        return None
    try:
        return datetime.date.fromisoformat(date_str)
    except Exception:
        return None


def select_relevant_activities(activities):
    """
    根據「今天」挑出最相關的活動：

    1. 先找「尚未結束」或「即將開始」的活動（end_date >= today）
    2. 依 start_date 由近到遠排序，取前 10 筆
    3. 如果完全沒有未來活動，就挑「最近已結束」的 10 筆（start_date 最接近今天）
    """
    today = datetime.date.today()
    print(f"[DEBUG] today={today.isoformat()}")

    upcoming = []     # (start_date, activity)
    past_recent = []  # (start_date, activity)

    for a in activities:
        s = _parse_date(a.get("start_date"))
        e = _parse_date(a.get("end_date")) or s
        if s is None:
            continue

        if e is not None and e >= today:
            # 還沒結束的活動（包含今天正在進行中）
            upcoming.append((s, a))
        else:
            # 已經結束的活動，之後當備選
            past_recent.append((s, a))

    upcoming.sort(key=lambda x: x[0])              # 越早開始越前面
    past_recent.sort(key=lambda x: x[0], reverse=True)  # 最近結束的在前面

    if upcoming:
        chosen = [a for _, a in upcoming[:10]]
        mode = "upcoming"
    else:
        chosen = [a for _, a in past_recent[:10]]
        mode = "past"

    print(f"[DEBUG] select_relevant_activities: mode={mode}, count={len(chosen)}")
    return chosen, mode, today


def fetch_activity_data(user_text: str) -> dict:
    """
    封裝成跟之前類似的介面，方便後面 LLM 使用。
    目前來源是本地 JSON 檔。
    並在這裡就做「時間篩選」，避免 LLM 去選到很久以前的活動。
    """
    data = load_activities_from_file()
    all_activities = data.get("activities", [])
    selected, mode, today = select_relevant_activities(all_activities)

    return {
        "query": user_text,
        "activities": selected,
        "last_updated": data.get("last_updated"),
        "source": "local_json",
        "selection_mode": mode,          # upcoming / past
        "today": today.isoformat(),
        "total_activities": len(all_activities),
        "error": data.get("error"),
    }


# === 直接用 HTTP 呼叫 OpenAI Chat Completions API ===
def ask_llm_about_activities(user_text: str, api_json: dict) -> str:
    """
    把「使用者問題 + 活動 JSON」丟給 LLM，
    請他用中文整理成適合 LINE 顯示的回答。
    （不使用 openai 套件，只用 urllib）
    """
    if not OPENAI_API_KEY:
        return "目前尚未設定 LLM API 金鑰，暫時無法幫你分析活動，只能原樣回覆唷 QAQ"

    activities = api_json.get("activities", [])
    last_updated = api_json.get("last_updated")
    selection_mode = api_json.get("selection_mode")
    today_str = api_json.get("today")

    # 給 LLM 的 JSON，避免太肥只塞已篩選過的活動
    api_text = json.dumps(
        {
            "today": today_str,
            "last_updated": last_updated,
            "selection_mode": selection_mode,
            "activities": activities,
        },
        ensure_ascii=False,
    )

    # 告訴 LLM「今天是幾號」以及我們已經做過一次「最近活動」篩選
    system_prompt = (
        f"今天日期是 {today_str}。你是一個台北活動小幫手，會根據本地活動 JSON 來回答使用者問題。\n"
        "- 系統已經先幫你依照日期篩選過活動，提供的 activities 陣列，"
        "是『最近尚未結束或即將舉辦』（selection_mode = 'upcoming'）"
        "或『最近已結束的活動』（selection_mode = 'past'）。\n"
        "- 每筆活動包含標題、類別、地點、行政區、起迄日期、票價與簡介等欄位。\n"
        "- 使用者可能會問：最近台北有什麼活動、這週末有什麼適合親子、免費活動等等。\n"
        "- 請在提供的 activities 中，挑出 1~3 個最符合問題的活動，依序列出：活動名稱、日期（起迄）、地點，必要時加上票價或簡短說明。\n"
        "- 如果 selection_mode 是 'past'，要先說明『目前本地資料都是已結束的活動』，再推薦幾個最近的活動。\n"
        "- 回答使用繁體中文，字數控制在 8~12 行內，適合 LINE 聊天視窗閱讀。"
    )

    user_prompt = f"""
使用者問題：
{user_text}

本地活動 JSON（系統已先依日期篩選，僅保留最近的數筆）：
{api_text}
"""

    payload = {
        "model": "gpt-4o-mini",  # 或你有權限的其他模型
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.4,
    }

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }
    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        print("[DEBUG] calling OpenAI...")
        # 給 8 秒，搭配 Lambda 15 秒 timeout
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            resp_body = resp.read().decode("utf-8")
        resp_json = json.loads(resp_body)
        content = resp_json["choices"][0]["message"]["content"]
        print("[DEBUG] OpenAI done")
        return content.strip()
    except Exception as e:
        # 如果 OpenAI 叫不到，不要讓 Lambda 整個 timeout，回一個備用訊息
        print(f"[ERROR] OpenAI call failed: {e}")
        if not activities:
            return (
                "目前暫時無法連線到活動小幫手 LLM，且本地活動資料載入失敗 QQ\n"
                "建議稍後再試一次。"
            )
        return (
            "目前暫時無法連線到活動小幫手 LLM QQ\n"
            "我先直接列出幾個本地活動 JSON 裡的活動給你參考：\n\n"
            + "\n".join(
                f"・{a.get('title')}（{a.get('start_date')} 起，{a.get('location')}）"
                for a in activities[:5]
            )
        )


# === Lambda 入口點 ===
def lambda_handler(event, context):
    """
    Lambda Proxy 事件處理
    支援：
    - GET /  → 健康檢查
    - 任意 POST → LINE Webhook
    """

    # 取得 path（Function URL rawPath 或 API Gateway path）
    path = event.get("rawPath") or event.get("path", "/")
    method = (
        event.get("requestContext", {}).get("http", {}).get("method")
        or event.get("httpMethod", "GET")
    )

    print(f"[DEBUG] method={method}, path={path}")

    # 1. 健康檢查
    if method == "GET" and path == "/":
        print("[DEBUG] health check")
        data = load_activities_from_file()
        selected, mode, today = select_relevant_activities(data.get("activities", []))
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(
                {
                    "status": "ok",
                    "message": "Taipei Activity Bot Lambda is running with local JSON + LLM + date filter",
                    "activity_count": len(data.get("activities", [])),
                    "last_updated": data.get("last_updated"),
                    "selected_count": len(selected),
                    "selection_mode": mode,
                    "today": today.isoformat(),
                },
                ensure_ascii=False,
            ),
        }

    # 2. 所有 POST 當作 LINE Webhook
    if method == "POST":
        print("[DEBUG] handle LINE webhook")

        body = event.get("body", "")
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode("utf-8")

        # 驗證簽名
        headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
        signature = headers.get("x-line-signature", "")
        print(f"[DEBUG] headers keys={list(headers.keys())}")

        if not verify_signature(CHANNEL_SECRET, body, signature):
            print("[ERROR] invalid signature")
            return {"statusCode": 403, "body": "Invalid signature"}

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            print("[ERROR] invalid json")
            return {"statusCode": 400, "body": "Invalid JSON"}

        events = data.get("events", [])
        print(f"[DEBUG] events raw={events}")

        for ev in events:
            if ev.get("type") != "message":
                continue
            if ev.get("message", {}).get("type") != "text":
                continue

            reply_token = ev.get("replyToken")
            user_text = ev.get("message", {}).get("text", "")

            print(f"[DEBUG] user_text={user_text}")

            # 從本地 JSON 取得活動資料（含時間篩選）
            activity_json = fetch_activity_data(user_text)
            # 丟給 LLM 分析
            reply_text = ask_llm_about_activities(user_text, activity_json)

            print(f"[DEBUG] reply_text={reply_text}")

            # 回 LINE
            reply_message(reply_token, reply_text)

        return {"statusCode": 200, "body": "OK"}

    # 3. 其他情況
    print("[DEBUG] not matched route")
    return {"statusCode": 404, "body": "Not Found"}
