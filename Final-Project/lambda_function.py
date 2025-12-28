# lambda_function.py
# Taipei Activity Bot (LINE Webhook + Scheduler + S3 Cache + Taipei Open API + OpenAI)
# Full overwrite version for Henry
#
# Key behaviors (as you requested):
# 1) If S3 JSON is missing/empty OR last_updated is not today -> auto refresh via Taipei Open API and write back to S3
# 2) Debug logs must show whether API is called, success, total/data counts, normalized counts, and S3 write counts
# 3) Keep Scheduled Event update (EventBridge) + keep LINE webhook + keep GET healthcheck
# 4) Taipei Open API response shape per your screenshot: {"total": N, "data": [ ... ]}

import os
import json
import hmac
import hashlib
import base64
import datetime
import urllib.parse
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

# =====================
# Module-level cache (reused across warm invocations)
# =====================
# NOTE:
# - Lambda 「同一個 container」在 warm 狀態時會重複使用這個全域變數
# - 目的：LINE 使用者每次問，不要都讀 S3（減少 latency）
# - 但注意：Scheduler 更新完 S3 後，一定要把它清掉（避免讀到舊 cache）
ACTIVITY_CACHE: Optional[Dict[str, Any]] = None

# =====================
# ENV
# =====================
# LINE
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")  # LINE Webhook 簽名驗證用
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")  # 回覆訊息用

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # 你沒設也可以，會 fallback 直接列活動
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # 模型可換

# Activity Source
# NOTE: 正式部署建議固定用 s3
ACTIVITY_SOURCE = os.getenv("ACTIVITY_SOURCE", "s3").lower()  # s3 / local

# S3
S3_BUCKET = os.getenv("S3_BUCKET", "")  # 放活動 JSON 的 bucket
S3_KEY = os.getenv("S3_KEY", "taipei/taipei_activities.json")  # JSON 物件路徑

# Taipei Open API
# NOTE: 你 curl 驗證過必須帶 begin/end/page
TAIPEI_API_BASE = "https://www.travel.taipei/open-api"
TAIPEI_API_LANG = os.getenv("TAIPEI_API_LANG", "zh-tw")  # zh-tw / en / ...
TAIPEI_API_DAYS_AHEAD = int(os.getenv("TAIPEI_API_DAYS_AHEAD", "60"))  # 往未來抓幾天
TAIPEI_API_DAYS_BACK = int(os.getenv("TAIPEI_API_DAYS_BACK", "30"))  # 往過去抓幾天（避免漏）

# Local fallback
LOCAL_ACTIVITY_JSON_FILE = "taipei_activities.json"  # ACTIVITY_SOURCE=local 用

# Optional: protect /refresh endpoint
# NOTE: 如果你不想讓任何人打 /refresh，就把 token 設起來
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN", "")  # if empty => no protection

# Auto-refresh rule: require today's data?
# NOTE:
# - True：只要 last_updated 不是今天，就 refresh（你說「清 S3 後理論上會自動補」）
# - False：只要 S3 有資料就用，不會每天 refresh
REQUIRE_TODAY = os.getenv("REQUIRE_TODAY", "1") == "1"  # default true


# =====================
# Helpers: LINE signature
# =====================
def verify_signature(channel_secret: str, body: str, signature: str) -> bool:
    # NOTE:
    # - LINE Webhook 的安全機制，必須用 CHANNEL_SECRET 驗證
    # - 若 signature 不符 → 直接拒絕（403）
    if not channel_secret or not signature:
        return False

    mac = hmac.new(
        channel_secret.encode("utf-8"),
        body.encode("utf-8"),
        hashlib.sha256,
    ).digest()

    expected = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expected, signature)


# =====================
# Helpers: LINE reply
# =====================
def reply_message(reply_token: str, text: str) -> None:
    # NOTE:
    # - LINE Reply API：必須在 webhook 回來後很短時間內回覆（通常 1 分鐘內）
    # - timeout 設短一點避免卡住
    if not CHANNEL_ACCESS_TOKEN:
        print("[ERROR] LINE_CHANNEL_ACCESS_TOKEN missing")
        return

    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}",
    }

    body = {
        "replyToken": reply_token,
        "messages": [{"type": "text", "text": text}],
    }

    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        # DEBUG: 這裡成功才會真的回覆到 LINE
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            resp.read()
        print("[DEBUG] LINE reply ok")
    except Exception as e:
        # NOTE: 回覆失敗不應該讓整個 Lambda 失敗，否則 LINE 會 retry
        print(f"[ERROR] LINE reply failed: {e}")


# =====================
# Date tools
# =====================
def _parse_date(date_str: Optional[str]) -> Optional[datetime.date]:
    # NOTE:
    # - 台北 API begin/end 可能是 "2025-12-06 11:00:00 +08:00"
    # - 我們只取前 10 碼 "YYYY-MM-DD"
    if not date_str:
        return None
    try:
        return datetime.date.fromisoformat(str(date_str)[:10])
    except Exception:
        return None


def _parse_last_updated_to_date(last_updated: Optional[str]) -> Optional[datetime.date]:
    # NOTE:
    # - last_updated 預期格式 "YYYY-MM-DDTHH:MM:SSZ"
    # - 同樣只取 YYYY-MM-DD
    if not last_updated:
        return None
    try:
        return datetime.date.fromisoformat(str(last_updated)[:10])
    except Exception:
        return None


def select_relevant_activities(
    activities: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], str, datetime.date]:
    """
    你原本的活動篩選規則（給 LLM 用）：
    1) pick end_date >= today (upcoming/ongoing)
    2) sort by start_date asc take 10
    3) if none, pick most recent past 10
    """
    today = datetime.date.today()
    upcoming: List[Tuple[datetime.date, Dict[str, Any]]] = []
    past_recent: List[Tuple[datetime.date, Dict[str, Any]]] = []

    for a in activities:
        # 解析 start/end
        s = _parse_date(a.get("start_date"))
        e = _parse_date(a.get("end_date")) or s

        # 沒 start_date 的直接丟掉（資料不完整）
        if s is None:
            continue

        # end_date >= today -> upcoming
        if e is not None and e >= today:
            upcoming.append((s, a))
        else:
            past_recent.append((s, a))

    upcoming.sort(key=lambda x: x[0])  # 越近越前面
    past_recent.sort(key=lambda x: x[0], reverse=True)  # 最近結束的優先

    if upcoming:
        chosen = [a for _, a in upcoming[:10]]
        mode = "upcoming"
    else:
        chosen = [a for _, a in past_recent[:10]]
        mode = "past"

    # DEBUG: 你在 CloudWatch 看這行就知道現在挑的是未來 or 過去資料
    print(f"[DEBUG] today={today.isoformat()}")
    print(f"[DEBUG] select_relevant_activities: mode={mode}, count={len(chosen)}")
    return chosen, mode, today


# =====================
# S3 SigV4 (standard lib only)
# =====================
# NOTE:
# - 你希望不用 boto3，所以用 SigV4 自己簽名
# - Lambda runtime 會自動注入 AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN
def _aws_sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _aws_get_signature_key(key: str, date_stamp: str, region: str, service: str) -> bytes:
    # SigV4 標準流程
    k_date = _aws_sign(("AWS4" + key).encode("utf-8"), date_stamp)
    k_region = hmac.new(k_date, region.encode("utf-8"), hashlib.sha256).digest()
    k_service = hmac.new(k_region, service.encode("utf-8"), hashlib.sha256).digest()
    k_signing = hmac.new(k_service, "aws4_request".encode("utf-8"), hashlib.sha256).digest()
    return k_signing


def s3_put_json(bucket: str, key: str, obj: Dict[str, Any]) -> None:
    # NOTE:
    # - PUT 到 private bucket 需要簽名
    # - 你的 Lambda role 必須有 s3:PutObject 權限

    access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    token = os.getenv("AWS_SESSION_TOKEN", "")
    region = os.getenv("AWS_REGION", "ap-northeast-1")

    if not access_key or not secret_key:
        raise RuntimeError("Missing AWS credentials in environment")

    service = "s3"
    host = f"{bucket}.s3.{region}.amazonaws.com"
    endpoint = f"https://{host}/{urllib.parse.quote(key)}"

    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    payload_hash = hashlib.sha256(payload).hexdigest()

    t = datetime.datetime.utcnow()
    amz_date = t.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = t.strftime("%Y%m%d")

    canonical_uri = "/" + "/".join([urllib.parse.quote(p) for p in key.split("/")])
    canonical_querystring = ""

    canonical_headers = (
        f"host:{host}\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        f"x-amz-date:{amz_date}\n"
    )
    signed_headers = "host;x-amz-content-sha256;x-amz-date"

    headers = {
        "Content-Type": "application/json",
        "x-amz-date": amz_date,
        "x-amz-content-sha256": payload_hash,
    }

    # NOTE: Lambda temporary credentials 會有 session token，必須簽進去
    if token:
        canonical_headers += f"x-amz-security-token:{token}\n"
        signed_headers += ";x-amz-security-token"
        headers["x-amz-security-token"] = token

    canonical_request = (
        "PUT\n"
        + canonical_uri
        + "\n"
        + canonical_querystring
        + "\n"
        + canonical_headers
        + "\n"
        + signed_headers
        + "\n"
        + payload_hash
    )

    algorithm = "AWS4-HMAC-SHA256"
    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = (
        algorithm
        + "\n"
        + amz_date
        + "\n"
        + credential_scope
        + "\n"
        + hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    )

    signing_key = _aws_get_signature_key(secret_key, date_stamp, region, service)
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    headers["Authorization"] = (
        f"{algorithm} Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )

    # DEBUG: 這行可以看 endpoint，排查 region/bucket/key 是否正確
    print(f"[DEBUG] s3_put_json endpoint={endpoint}")

    req = urllib.request.Request(endpoint, data=payload, headers=headers, method="PUT")
    with urllib.request.urlopen(req, timeout=12.0) as resp:
        resp.read()
    # DEBUG: 寫成功才會跑到這裡
    print("[DEBUG] s3_put_json done")


def s3_get_json(bucket: str, key: str) -> Dict[str, Any]:
    # NOTE:
    # - S3 GET 同樣需要 SigV4（private bucket）
    # - 若 object 不存在，會噴 urllib.error.HTTPError(404)

    access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    token = os.getenv("AWS_SESSION_TOKEN", "")
    region = os.getenv("AWS_REGION", "ap-northeast-1")

    if not access_key or not secret_key:
        raise RuntimeError("Missing AWS credentials in environment")

    service = "s3"
    host = f"{bucket}.s3.{region}.amazonaws.com"
    endpoint = f"https://{host}/{urllib.parse.quote(key)}"

    payload_hash = hashlib.sha256(b"").hexdigest()

    t = datetime.datetime.utcnow()
    amz_date = t.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = t.strftime("%Y%m%d")

    canonical_uri = "/" + "/".join([urllib.parse.quote(p) for p in key.split("/")])
    canonical_querystring = ""

    canonical_headers = (
        f"host:{host}\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        f"x-amz-date:{amz_date}\n"
    )
    signed_headers = "host;x-amz-content-sha256;x-amz-date"

    headers = {
        "x-amz-date": amz_date,
        "x-amz-content-sha256": payload_hash,
    }

    if token:
        canonical_headers += f"x-amz-security-token:{token}\n"
        signed_headers += ";x-amz-security-token"
        headers["x-amz-security-token"] = token

    canonical_request = (
        "GET\n"
        + canonical_uri
        + "\n"
        + canonical_querystring
        + "\n"
        + canonical_headers
        + "\n"
        + signed_headers
        + "\n"
        + payload_hash
    )

    algorithm = "AWS4-HMAC-SHA256"
    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = (
        algorithm
        + "\n"
        + amz_date
        + "\n"
        + credential_scope
        + "\n"
        + hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    )

    signing_key = _aws_get_signature_key(secret_key, date_stamp, region, service)
    signature = hmac.new(signing_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()

    headers["Authorization"] = (
        f"{algorithm} Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )

    # DEBUG: 你在 CloudWatch 看這行就知道有沒有真的去讀 S3
    print(f"[DEBUG] s3_get_json endpoint={endpoint}")

    req = urllib.request.Request(endpoint, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=12.0) as resp:
        body = resp.read().decode("utf-8")

    # NOTE: 如果 S3 內容不是 JSON，json.loads 會噴錯
    return json.loads(body)


# =====================
# Taipei Open API fetch (supports {"total":N,"data":[...]})
# =====================
def fetch_taipei_activities_from_api(lang: str, begin: str, end: str) -> Tuple[List[Dict[str, Any]], int]:
    # NOTE:
    # - 這段是你最在意「是否有照規則帶 begin/end」的地方
    # - 每一頁都用 begin/end/page
    # - 回傳格式依你的截圖：{"total": N, "data": [...]}

    results: List[Dict[str, Any]] = []
    page = 1
    total_reported = 0

    while True:
        # 這裡確保 begin/end/page 都會帶上，符合你 curl 規則
        qs = urllib.parse.urlencode({"begin": begin, "end": end, "page": page})
        url = f"{TAIPEI_API_BASE}/{lang}/Events/Activity?{qs}"

        req = urllib.request.Request(
            url,
            headers={"accept": "application/json", "User-Agent": "taipei-activity-bot/1.0"},
            method="GET",
        )

        # DEBUG: 你要看「是否真的打 API」就看這行
        print(f"[DEBUG] taipei_api_request: page={page}, url={url}")

        try:
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                status = getattr(resp, "status", 200)  # urllib 版本差異，保險用 getattr
                body = resp.read().decode("utf-8")
            # DEBUG: status/body_len 用來判斷 API 成功且回應不是空
            print(f"[DEBUG] taipei_api_response: status={status}, body_len={len(body)}")
        except Exception as e:
            # NOTE: API 失敗就 break，避免無限 loop
            print(f"[ERROR] taipei api request failed: {e}")
            break

        try:
            data = json.loads(body)
        except Exception as e:
            # NOTE: 有時候 API 回 HTML（例如 502），這時 json.loads 會失敗
            print(f"[ERROR] taipei api json decode failed: {e}")
            print(f"[DEBUG] body_head={body[:500]}")
            break

        # ✅ 依你的 API 圖：dict + total + data(list)
        if isinstance(data, dict):
            if isinstance(data.get("total"), int):
                total_reported = data.get("total") or total_reported  # 第一頁通常才會有 total
            items = data.get("data")
            if not isinstance(items, list):
                # DEBUG: 回應結構不符合預期時，把 key 印出來方便你對照 API 截圖
                print("[WARN] taipei api dict but 'data' is not list")
                print(f"[DEBUG] dict keys={list(data.keys())}")
                print(f"[DEBUG] body_head={body[:500]}")
                break
        elif isinstance(data, list):
            # NOTE: 保險：如果 API 某天直接回 list 也能跑
            items = data
        else:
            print(f"[WARN] unexpected taipei api response type={type(data)}")
            print(f"[DEBUG] body_head={body[:500]}")
            break

        # DEBUG: 你要看每頁實際拿到幾筆，就看這行
        print(f"[DEBUG] taipei_api_page_items: page={page}, total_reported={total_reported}, items_len={len(items)}")

        if len(items) == 0:
            # NOTE: 沒資料代表到底了
            break

        # NOTE: 只保留 dict item，避免混入 None/str 造成 normalize 崩
        results.extend([x for x in items if isinstance(x, dict)])

        # NOTE: 台北 API page size 通常 30；小於 30 表示最後一頁
        if len(items) < 30:
            break

        page += 1
        if page > 200:
            # NOTE: 防止 API 異常造成無限分頁
            print("[WARN] page > 200, stop")
            break

    # DEBUG: 你要看總共抓了多少筆，就看這行
    print(f"[DEBUG] taipei_api_fetched_total_items={len(results)}, total_reported={total_reported}")
    return results, total_reported


def normalize_activity(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize fields for your bot, based on your API screenshot.
    Example fields: title, description, begin, end, address, district, ticket, ...
    """
    # NOTE:
    # - 這裡是欄位映射：把台北 API 欄位轉成你 bot 固定欄位（start_date/end_date/title...）
    # - raw 保留原始 item，方便你之後排查欄位不一致

    title = item.get("title") or item.get("name") or ""  # 活動名稱
    intro = item.get("description") or item.get("introduction") or item.get("content") or ""  # 簡介

    begin = item.get("begin") or item.get("start") or item.get("startDate") or item.get("start_date")  # 起始
    end = item.get("end") or item.get("endDate") or item.get("end_date")  # 結束

    location = item.get("address") or item.get("location") or item.get("place") or ""  # 地點/地址
    district = item.get("district") or item.get("area") or ""  # 行政區
    ticket = item.get("ticket") or item.get("price") or ""  # 票價/費用

    def to_yyyy_mm_dd(v: Any) -> Optional[str]:
        # NOTE:
        # - begin/end 可能是 "2025-12-06 00:00:00 +08:00"
        # - 一律只取 "YYYY-MM-DD"
        if not v:
            return None
        s = str(v)
        return s[:10] if len(s) >= 10 else None

    return {
        "title": title,
        "category": item.get("category") or item.get("type") or "",
        "location": location,
        "district": district,
        "start_date": to_yyyy_mm_dd(begin),
        "end_date": to_yyyy_mm_dd(end),
        "price": ticket,
        "intro": intro,
        "raw": item,  # DEBUG: 保留原始資料方便你確認欄位
    }


def build_activity_json_for_storage(api_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    # NOTE:
    # - 這裡做 normalize + 丟掉不完整資料
    # - 丟掉規則：沒有 start_date 或 title 的視為垃圾資料

    normalized = [normalize_activity(x) for x in api_items]

    before = len(normalized)
    normalized = [x for x in normalized if x.get("start_date") and x.get("title")]
    after = len(normalized)

    # DEBUG: 你要看 normalize 前後差多少，就看這行
    print(f"[DEBUG] normalize: before={before}, after_drop_invalid={after}")

    return {
        # NOTE: last_updated 使用 UTC，方便和 Lambda/CloudWatch 對齊
        "last_updated": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "activities": normalized,
    }


# =====================
# Auto refresh logic
# =====================
def needs_refresh(data: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Decide whether we must call Taipei API.
    Conditions:
    - activities missing or empty
    - REQUIRE_TODAY=1 and last_updated date != today
    """
    today = datetime.date.today()
    activities = data.get("activities")

    # 1) S3 沒資料 / 被你清空 / 結構不對 → 一律 refresh
    if not isinstance(activities, list) or len(activities) == 0:
        return True, "activities_empty"

    # 2) 若你要求「一定要今天」 → last_updated 不是今天就 refresh
    if REQUIRE_TODAY:
        last_updated = data.get("last_updated")
        lu_date = _parse_last_updated_to_date(last_updated)
        if lu_date is None:
            return True, "last_updated_missing_or_invalid"
        if lu_date != today:
            return True, f"last_updated_not_today({lu_date.isoformat()} != {today.isoformat()})"

    return False, "ok"


def refresh_from_api_and_persist() -> Dict[str, Any]:
    """
    Fetch Taipei API and write to S3, then clear cache.
    """
    global ACTIVITY_CACHE

    today = datetime.date.today()

    # NOTE:
    # - begin/end 按你的規則計算
    # - begin 往回抓，避免漏掉正在進行的活動
    # - end 往前抓，提供未來活動推薦
    begin = (today - datetime.timedelta(days=TAIPEI_API_DAYS_BACK)).isoformat()
    end = (today + datetime.timedelta(days=TAIPEI_API_DAYS_AHEAD)).isoformat()
    lang = TAIPEI_API_LANG

    # DEBUG: 你要確認 refresh 的 begin/end 正確，就看這行
    print(f"[DEBUG] refresh_start: begin={begin}, end={end}, lang={lang}")

    # ✅ 真的打 Taipei API 就在這行（你的 delay 也主要在這裡）
    api_items, total_reported = fetch_taipei_activities_from_api(lang=lang, begin=begin, end=end)

    # ✅ normalize + 組出要存到 S3 的 JSON
    storage_obj = build_activity_json_for_storage(api_items)

    # DEBUG: 你說以前有的「成功筆數」都在這行
    print(
        f"[DEBUG] refresh_counts: total_reported={total_reported}, "
        f"api_items={len(api_items)}, normalized={len(storage_obj.get('activities', []))}"
    )

    # 只有 ACTIVITY_SOURCE=s3 才能把資料 persist
    if ACTIVITY_SOURCE != "s3":
        raise RuntimeError("refresh_from_api requires ACTIVITY_SOURCE=s3 to persist")

    # 沒 bucket 直接噴錯，避免 silent fail
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET is empty")

    # DEBUG: 你要確認寫到哪裡，就看這行
    print(f"[DEBUG] writing_to_s3: s3://{S3_BUCKET}/{S3_KEY}")

    # ✅ 寫入 S3（若權限或簽名錯誤會在這裡爆）
    s3_put_json(S3_BUCKET, S3_KEY, storage_obj)
    print("[DEBUG] s3_write_done")

    # NOTE:
    # - 這行非常重要：清掉 cache，確保下一次 load 會去 S3 拿最新
    ACTIVITY_CACHE = None

    return {
        "ok": True,
        "begin": begin,
        "end": end,
        "lang": lang,
        "stored_to": f"s3://{S3_BUCKET}/{S3_KEY}",
        "total_reported": total_reported,
        "api_items": len(api_items),
        "normalized": len(storage_obj.get("activities", [])),
        "last_updated": storage_obj.get("last_updated"),
    }


# =====================
# Load activities (S3/local) + cache + auto refresh
# =====================
def load_activities(auto_refresh: bool = True) -> Dict[str, Any]:
    """
    Unified loader with module cache.
    If auto_refresh is True and source is S3, it will:
      - load S3
      - check empty/outdated
      - refresh from API if needed
      - reload S3
    """
    global ACTIVITY_CACHE

    # NOTE:
    # - 若 cache 有值，直接回傳（這是 LINE 問答速度快的關鍵）
    if ACTIVITY_CACHE is not None:
        print("[DEBUG] load_activities: hit module cache")
        return ACTIVITY_CACHE

    # 1) Load raw data from S3/local
    try:
        if ACTIVITY_SOURCE == "s3":
            if not S3_BUCKET:
                raise ValueError("S3_BUCKET is empty but ACTIVITY_SOURCE=s3")

            # DEBUG: 你要知道是否真的去讀 S3，就看這行
            print(f"[DEBUG] loading_from_s3: s3://{S3_BUCKET}/{S3_KEY}")

            data = s3_get_json(S3_BUCKET, S3_KEY)
        else:
            # local fallback
            file_path = os.path.join(os.path.dirname(__file__), LOCAL_ACTIVITY_JSON_FILE)
            print(f"[DEBUG] loading_from_local: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

    except urllib.error.HTTPError as e:
        # NOTE:
        # - 這裡最常見：你把 S3 清掉了 → GET 會 404
        # - 你希望這時候會 refresh，所以回傳空 activities 讓 needs_refresh=True
        print(f"[WARN] s3_get_json http error: {e}")
        data = {"last_updated": None, "activities": [], "error": f"s3_http_error:{e.code}"}

    except Exception as e:
        # NOTE:
        # - 其它錯誤：例如簽名錯誤、權限錯誤、JSON parse 錯
        print(f"[ERROR] load_activities initial load failed: {e}")
        data = {"last_updated": None, "activities": [], "error": str(e)}

    # 形狀防呆：確保一定有 dict + activities(list)
    if not isinstance(data, dict):
        data = {"last_updated": None, "activities": [], "error": "invalid_s3_json_not_dict"}
    if "activities" not in data or not isinstance(data.get("activities"), list):
        data["activities"] = []
        data["error"] = data.get("error") or "invalid_s3_json_structure"

    # DEBUG: 你要看「現在 S3 讀到多少筆」就看這行
    print(
        f"[DEBUG] loaded_snapshot: activities={len(data.get('activities', []))}, "
        f"last_updated={data.get('last_updated')}, error={data.get('error')}"
    )

    # 2) Auto refresh if needed
    # NOTE:
    # - 這就是你要求的：「S3 空/不是今天」要自動 refresh
    # - 你也說過：「最好只有 Scheduler 打 API」，但你同時又要求「S3 空要自動補」
    # - 所以這裡會在必要時 refresh（第一次 S3 空時）
    if auto_refresh and ACTIVITY_SOURCE == "s3":
        need, reason = needs_refresh(data)

        # DEBUG: 你要看「為什麼要 refresh」就看這行
        print(f"[DEBUG] needs_refresh={need}, reason={reason}")

        if need:
            try:
                # ✅ 這裡會真的打 API 並寫回 S3
                refresh_result = refresh_from_api_and_persist()

                # DEBUG: refresh 的回傳 summary（含筆數）
                print(f"[DEBUG] refresh_result: {json.dumps(refresh_result, ensure_ascii=False)}")

                # ✅ refresh 完再讀一次 S3，確保回給 LINE 的是最新的
                print(f"[DEBUG] reloading_from_s3_after_refresh: s3://{S3_BUCKET}/{S3_KEY}")
                data = s3_get_json(S3_BUCKET, S3_KEY)

                # DEBUG: reload 後 S3 真的有資料了才會看到這行 >0
                print(
                    f"[DEBUG] reload_after_refresh: activities={len(data.get('activities', []))}, "
                    f"last_updated={data.get('last_updated')}"
                )

            except Exception as e:
                # NOTE:
                # - refresh 失敗時，不要讓整個流程崩掉（LINE 還是要回應）
                # - error 記在 data 裡，讓 healthcheck / LINE 能看到
                print(f"[ERROR] auto refresh failed: {e}")
                data["error"] = f"auto_refresh_failed:{str(e)}"

    # 3) cache and return
    # NOTE:
    # - 這行會讓後續 LINE 問答更快（不重讀 S3）
    ACTIVITY_CACHE = data
    return data


# =====================
# Activity interface for LLM
# =====================
def fetch_activity_data(user_text: str) -> Dict[str, Any]:
    # NOTE:
    # - 這裡會 load S3 並視需要 auto_refresh
    # - LINE delay 的來源之一就是這裡（當 need refresh 時）
    data = load_activities(auto_refresh=True)

    all_activities = data.get("activities", [])

    # NOTE:
    # - 這裡是把「所有活動」縮小成「最近/即將」的 10 筆給 LLM
    selected, mode, today = select_relevant_activities(all_activities)

    return {
        "query": user_text,
        "activities": selected,
        "last_updated": data.get("last_updated"),
        "source": ACTIVITY_SOURCE,
        "selection_mode": mode,
        "today": today.isoformat(),
        "total_activities": len(all_activities),
        "error": data.get("error"),
    }


# =====================
# OpenAI (kept compatible with your original)
# =====================
def ask_llm_about_activities(user_text: str, api_json: Dict[str, Any]) -> str:
    activities = api_json.get("activities", [])
    last_updated = api_json.get("last_updated")
    selection_mode = api_json.get("selection_mode")
    today_str = api_json.get("today")

    # NOTE:
    # - 沒 OpenAI key：直接列活動（避免你 debug 時還卡在 LLM）
    if not OPENAI_API_KEY:
        if not activities:
            return "目前活動資料為空（S3 可能尚未更新成功）。\n你可以稍後再試或觸發更新。"
        return "（未設定 OpenAI 金鑰，先給你活動清單）\n" + "\n".join(
            f"・{a.get('title')}（{a.get('start_date')}~{a.get('end_date') or a.get('start_date')}，{a.get('location')}）"
            for a in activities[:6]
        )

    # NOTE:
    # - 把你挑過的活動 JSON 丟給 LLM
    api_text = json.dumps(
        {
            "today": today_str,
            "last_updated": last_updated,
            "selection_mode": selection_mode,
            "activities": activities,
        },
        ensure_ascii=False,
    )

    system_prompt = (
        f"今天日期是 {today_str}。你是一個台北活動小幫手，會根據活動 JSON 回答使用者。\n"
        "- 只從 JSON 內容回答，不要編造。\n"
        "- 請挑出 1~3 個最符合問題的活動，列：活動名稱、日期（起迄）、地點，必要時加票價或一句簡介。\n"
        "- 若 selection_mode='past'，先提醒目前資料多為已結束活動。\n"
        "- 用繁體中文，控制在 8~12 行內，適合 LINE 閱讀。"
    )

    user_prompt = f"使用者問題：\n{user_text}\n\n活動 JSON（已篩選過最近數筆）：\n{api_text}\n"

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.4,
    }

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        # DEBUG: LINE delay 的第二來源：OpenAI 呼叫時間
        print("[DEBUG] calling OpenAI...")

        with urllib.request.urlopen(req, timeout=15.0) as resp:
            resp_body = resp.read().decode("utf-8")

        resp_json = json.loads(resp_body)
        content = resp_json["choices"][0]["message"]["content"]

        print("[DEBUG] OpenAI done")
        return (content or "").strip()

    except Exception as e:
        # NOTE:
        # - OpenAI 失敗時 fallback 列活動，不要讓 LINE 沒回
        print(f"[ERROR] OpenAI call failed: {e}")

        if not activities:
            return "目前暫時無法連線 LLM，且活動資料為空。\n請稍後再試或先觸發更新。"

        return (
            "LLM 暫時連不上，我先列幾個活動給你：\n"
            + "\n".join(
                f"・{a.get('title')}（{a.get('start_date')}，{a.get('location')}）"
                for a in activities[:6]
            )
        )


# =====================
# Event detection (DON'T mis-classify HTTP GET / as scheduled)
# =====================
def is_scheduled_event(event: Dict[str, Any]) -> bool:
    # NOTE:
    # - 避免誤判：有些平台會把 http event 包成 dict，看起來像有 source
    # - 只認 EventBridge 兩種明確特徵
    if event.get("source") == "aws.events":
        return True
    if event.get("detail-type") in ("Scheduled Event", "EventBridge Scheduler"):
        return True
    return False


# =====================
# Lambda handler
# =====================
def lambda_handler(event, context):
    # DEBUG: 每次 invocation 第一行先印 event keys，方便你排查是 HTTP 還是 Scheduler
    try:
        print(f"[DEBUG] event_keys={list(event.keys())}")
    except Exception:
        print("[DEBUG] event_keys=unknown")

    # ---- Scheduled update (EventBridge) ----
    # NOTE:
    # - 你要求「Scheduler 才走 API」，所以這裡一定會 refresh_from_api_and_persist()
    if isinstance(event, dict) and is_scheduled_event(event):
        print("[DEBUG] scheduled event triggered")
        try:
            result = refresh_from_api_and_persist()
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(result, ensure_ascii=False),
            }
        except Exception as e:
            print(f"[ERROR] scheduled update failed: {e}")
            return {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False),
            }

    # ---- HTTP routing ----
    # NOTE:
    # - 支援 Lambda Function URL / API Gateway
    # - Function URL: rawPath + requestContext.http.method
    # - API Gateway v1: path + httpMethod
    path = event.get("rawPath") or event.get("path", "/")
    method = (
        event.get("requestContext", {}).get("http", {}).get("method")
        or event.get("httpMethod", "GET")
    )
    print(f"[DEBUG] method={method}, path={path}")

    # GET /
    # NOTE:
    # - 健康檢查用
    # - 也會走 load_activities(auto_refresh=True)（你希望清空後能自動補）
    if method == "GET" and path == "/":
        data = load_activities(auto_refresh=True)  # ✅ 你期望的：空/非今日會自動 refresh
        selected, mode, today = select_relevant_activities(data.get("activities", []))
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(
                {
                    "status": "ok",
                    "message": "Taipei Activity Bot running",
                    "source": ACTIVITY_SOURCE,
                    "activity_count": len(data.get("activities", [])),
                    "last_updated": data.get("last_updated"),
                    "selected_count": len(selected),
                    "selection_mode": mode,
                    "today": today.isoformat(),
                    "error": data.get("error"),
                    "s3_bucket": S3_BUCKET,
                    "s3_key": S3_KEY,
                    "require_today": REQUIRE_TODAY,
                },
                ensure_ascii=False,
            ),
        }

    # GET /refresh (manual force refresh)
    # NOTE:
    # - 手動強制打 API + 寫 S3
    # - 可用 token 保護，避免被亂刷 API
    if method == "GET" and path == "/refresh":
        qs = event.get("rawQueryString") or ""
        params = urllib.parse.parse_qs(qs)
        token = (params.get("token") or [""])[0]

        if REFRESH_TOKEN and token != REFRESH_TOKEN:
            # DEBUG: token 不對就拒絕
            return {"statusCode": 403, "body": "Forbidden"}

        try:
            result = refresh_from_api_and_persist()
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(result, ensure_ascii=False),
            }
        except Exception as e:
            return {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False),
            }

    # POST (LINE Webhook)
    # NOTE:
    # - LINE 只會打 POST
    # - 這裡會做簽名驗證，並且逐個 event 處理
    if method == "POST":
        body = event.get("body", "") or ""

        # NOTE:
        # - API Gateway 可能 base64 encoded
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode("utf-8")

        headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
        signature = headers.get("x-line-signature", "")

        # ✅ LINE 安全驗證
        if not verify_signature(CHANNEL_SECRET, body, signature):
            print("[ERROR] invalid signature")
            return {"statusCode": 403, "body": "Invalid signature"}

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return {"statusCode": 400, "body": "Invalid JSON"}

        events = data.get("events", [])

        # NOTE:
        # - LINE 一次 webhook 可能帶多個 events
        for ev in events:
            if ev.get("type") != "message":
                continue
            if ev.get("message", {}).get("type") != "text":
                continue

            reply_token = ev.get("replyToken")
            user_text = ev.get("message", {}).get("text", "").strip()

            if not reply_token or not user_text:
                continue

            # ✅ 你要的重點：LINE 正常情況只讀 S3（快）
            # 但若 S3 空/非今日，load_activities(auto_refresh=True) 會觸發 refresh（慢）
            # DEBUG: 你在 CloudWatch 看 needs_refresh log 就知道有沒有打 API
            activity_json = fetch_activity_data(user_text)

            # ✅ LLM 回答（若沒 key 就 fallback）
            reply_text = ask_llm_about_activities(user_text, activity_json)

            # ✅ 回覆 LINE
            reply_message(reply_token, reply_text)

        return {"statusCode": 200, "body": "OK"}

    # 其它 path
    return {"statusCode": 404, "body": "Not Found"}
