# lambda_function.py
# Taipei Activity Bot (LINE Webhook + Scheduler + S3 Cache + Taipei Open API + OpenAI)
# Full overwrite version for Henry
#
# Key behaviors (as you requested):
# 1) If S3 JSON is missing/empty OR last_updated is not today -> auto refresh via Taipei Open API and write back to S3
# 2) Debug logs must show whether API is called, success, total/data counts, normalized counts, and S3 write counts
# 3) Keep Scheduled Event update (EventBridge) + keep LINE webhook + keep GET healthcheck
# 4) Taipei Open API response shape per your screenshot: {"total": N, "data": [ ... ]} fff

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
ACTIVITY_CACHE: Optional[Dict[str, Any]] = None

# =====================
# ENV
# =====================
# LINE
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET", "")
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Activity Source
ACTIVITY_SOURCE = os.getenv("ACTIVITY_SOURCE", "s3").lower()  # s3 / local

# S3
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_KEY = os.getenv("S3_KEY", "taipei/taipei_activities.json")

# Taipei Open API
TAIPEI_API_BASE = "https://www.travel.taipei/open-api"
TAIPEI_API_LANG = os.getenv("TAIPEI_API_LANG", "zh-tw")
TAIPEI_API_DAYS_AHEAD = int(os.getenv("TAIPEI_API_DAYS_AHEAD", "60"))
TAIPEI_API_DAYS_BACK = int(os.getenv("TAIPEI_API_DAYS_BACK", "30"))

# Local fallback
LOCAL_ACTIVITY_JSON_FILE = "taipei_activities.json"

# Optional: protect /refresh endpoint
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN", "")  # if empty => no protection

# Auto-refresh rule: require today's data?
REQUIRE_TODAY = os.getenv("REQUIRE_TODAY", "1") == "1"  # default true


# =====================
# Helpers: LINE signature
# =====================
def verify_signature(channel_secret: str, body: str, signature: str) -> bool:
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
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            resp.read()
        print("[DEBUG] LINE reply ok")
    except Exception as e:
        print(f"[ERROR] LINE reply failed: {e}")


# =====================
# Date tools
# =====================
def _parse_date(date_str: Optional[str]) -> Optional[datetime.date]:
    if not date_str:
        return None
    try:
        # API returns like "2025-12-06 11:00:00 +08:00" or "2025-12-06"
        return datetime.date.fromisoformat(str(date_str)[:10])
    except Exception:
        return None


def _parse_last_updated_to_date(last_updated: Optional[str]) -> Optional[datetime.date]:
    if not last_updated:
        return None
    # expected: "YYYY-MM-DDTHH:MM:SSZ"
    try:
        return datetime.date.fromisoformat(str(last_updated)[:10])
    except Exception:
        return None


def select_relevant_activities(activities: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str, datetime.date]:
    """
    1) pick end_date >= today (upcoming/ongoing)
    2) sort by start_date asc take 10
    3) if none, pick most recent past 10
    """
    today = datetime.date.today()
    upcoming: List[Tuple[datetime.date, Dict[str, Any]]] = []
    past_recent: List[Tuple[datetime.date, Dict[str, Any]]] = []

    for a in activities:
        s = _parse_date(a.get("start_date"))
        e = _parse_date(a.get("end_date")) or s
        if s is None:
            continue
        if e is not None and e >= today:
            upcoming.append((s, a))
        else:
            past_recent.append((s, a))

    upcoming.sort(key=lambda x: x[0])
    past_recent.sort(key=lambda x: x[0], reverse=True)

    if upcoming:
        chosen = [a for _, a in upcoming[:10]]
        mode = "upcoming"
    else:
        chosen = [a for _, a in past_recent[:10]]
        mode = "past"

    print(f"[DEBUG] today={today.isoformat()}")
    print(f"[DEBUG] select_relevant_activities: mode={mode}, count={len(chosen)}")
    return chosen, mode, today


# =====================
# S3 SigV4 (standard lib only)
# =====================
def _aws_sign(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _aws_get_signature_key(key: str, date_stamp: str, region: str, service: str) -> bytes:
    k_date = _aws_sign(("AWS4" + key).encode("utf-8"), date_stamp)
    k_region = hmac.new(k_date, region.encode("utf-8"), hashlib.sha256).digest()
    k_service = hmac.new(k_region, service.encode("utf-8"), hashlib.sha256).digest()
    k_signing = hmac.new(k_service, "aws4_request".encode("utf-8"), hashlib.sha256).digest()
    return k_signing


def s3_put_json(bucket: str, key: str, obj: Dict[str, Any]) -> None:
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

    req = urllib.request.Request(endpoint, data=payload, headers=headers, method="PUT")
    with urllib.request.urlopen(req, timeout=12.0) as resp:
        resp.read()


def s3_get_json(bucket: str, key: str) -> Dict[str, Any]:
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

    req = urllib.request.Request(endpoint, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=12.0) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


# =====================
# Taipei Open API fetch (supports {"total":N,"data":[...]})
# =====================
def fetch_taipei_activities_from_api(lang: str, begin: str, end: str) -> Tuple[List[Dict[str, Any]], int]:
    results: List[Dict[str, Any]] = []
    page = 1
    total_reported = 0

    while True:
        qs = urllib.parse.urlencode({"begin": begin, "end": end, "page": page})
        url = f"{TAIPEI_API_BASE}/{lang}/Events/Activity?{qs}"

        req = urllib.request.Request(
            url,
            headers={"accept": "application/json", "User-Agent": "taipei-activity-bot/1.0"},
            method="GET",
        )

        print(f"[DEBUG] taipei_api_request: page={page}, url={url}")

        try:
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                status = getattr(resp, "status", 200)
                body = resp.read().decode("utf-8")
            print(f"[DEBUG] taipei_api_response: status={status}, body_len={len(body)}")
        except Exception as e:
            print(f"[ERROR] taipei api request failed: {e}")
            break

        try:
            data = json.loads(body)
        except Exception as e:
            print(f"[ERROR] taipei api json decode failed: {e}")
            print(f"[DEBUG] body_head={body[:500]}")
            break

        # Per your screenshot: {"total":24,"data":[...]}
        if isinstance(data, dict):
            if isinstance(data.get("total"), int):
                total_reported = data.get("total") or total_reported
            items = data.get("data")
            if not isinstance(items, list):
                print("[WARN] taipei api dict but 'data' is not list")
                print(f"[DEBUG] dict keys={list(data.keys())}")
                print(f"[DEBUG] body_head={body[:500]}")
                break
        elif isinstance(data, list):
            items = data
        else:
            print(f"[WARN] unexpected taipei api response type={type(data)}")
            print(f"[DEBUG] body_head={body[:500]}")
            break

        print(f"[DEBUG] taipei_api_page_items: page={page}, total_reported={total_reported}, items_len={len(items)}")

        if len(items) == 0:
            break

        # keep dict items only
        results.extend([x for x in items if isinstance(x, dict)])

        # Taipei API page size commonly 30
        if len(items) < 30:
            break

        page += 1
        if page > 200:
            print("[WARN] page > 200, stop")
            break

    print(f"[DEBUG] taipei_api_fetched_total_items={len(results)}, total_reported={total_reported}")
    return results, total_reported


def normalize_activity(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize fields for your bot, based on your API screenshot.
    Example fields: title, description, begin, end, address, district, ticket, ...
    """
    title = item.get("title") or item.get("name") or ""
    intro = item.get("description") or item.get("introduction") or item.get("content") or ""

    begin = item.get("begin") or item.get("start") or item.get("startDate") or item.get("start_date")
    end = item.get("end") or item.get("endDate") or item.get("end_date")

    location = item.get("address") or item.get("location") or item.get("place") or ""
    district = item.get("district") or item.get("area") or ""
    ticket = item.get("ticket") or item.get("price") or ""

    def to_yyyy_mm_dd(v: Any) -> Optional[str]:
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
        "raw": item,
    }


def build_activity_json_for_storage(api_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = [normalize_activity(x) for x in api_items]

    # Drop garbage: missing start_date or title
    before = len(normalized)
    normalized = [x for x in normalized if x.get("start_date") and x.get("title")]
    after = len(normalized)

    print(f"[DEBUG] normalize: before={before}, after_drop_invalid={after}")

    return {
        "last_updated": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "activities": normalized,
    }


# =====================
# Auto refresh logic (the part you complained was missing)
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
    if not isinstance(activities, list) or len(activities) == 0:
        return True, "activities_empty"

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
    begin = (today - datetime.timedelta(days=TAIPEI_API_DAYS_BACK)).isoformat()
    end = (today + datetime.timedelta(days=TAIPEI_API_DAYS_AHEAD)).isoformat()
    lang = TAIPEI_API_LANG

    print(f"[DEBUG] refresh_start: begin={begin}, end={end}, lang={lang}")

    api_items, total_reported = fetch_taipei_activities_from_api(lang=lang, begin=begin, end=end)
    storage_obj = build_activity_json_for_storage(api_items)

    # Debug counts (you said these used to exist)
    print(f"[DEBUG] refresh_counts: total_reported={total_reported}, api_items={len(api_items)}, normalized={len(storage_obj.get('activities', []))}")

    if ACTIVITY_SOURCE != "s3":
        raise RuntimeError("refresh_from_api requires ACTIVITY_SOURCE=s3 to persist")

    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET is empty")

    print(f"[DEBUG] writing_to_s3: s3://{S3_BUCKET}/{S3_KEY}")
    s3_put_json(S3_BUCKET, S3_KEY, storage_obj)
    print("[DEBUG] s3_write_done")

    # Clear cache so next load reads fresh S3
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

    if ACTIVITY_CACHE is not None:
        return ACTIVITY_CACHE

    # 1) Load raw data
    try:
        if ACTIVITY_SOURCE == "s3":
            if not S3_BUCKET:
                raise ValueError("S3_BUCKET is empty but ACTIVITY_SOURCE=s3")
            print(f"[DEBUG] loading_from_s3: s3://{S3_BUCKET}/{S3_KEY}")
            data = s3_get_json(S3_BUCKET, S3_KEY)
        else:
            file_path = os.path.join(os.path.dirname(__file__), LOCAL_ACTIVITY_JSON_FILE)
            print(f"[DEBUG] loading_from_local: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
    except urllib.error.HTTPError as e:
        # S3 object missing (404) should trigger refresh if allowed
        print(f"[WARN] s3_get_json http error: {e}")
        data = {"last_updated": None, "activities": [], "error": f"s3_http_error:{e.code}"}
    except Exception as e:
        print(f"[ERROR] load_activities initial load failed: {e}")
        data = {"last_updated": None, "activities": [], "error": str(e)}

    # Normalize shape guard
    if not isinstance(data, dict):
        data = {"last_updated": None, "activities": [], "error": "invalid_s3_json_not_dict"}
    if "activities" not in data or not isinstance(data.get("activities"), list):
        data["activities"] = []
        data["error"] = data.get("error") or "invalid_s3_json_structure"

    print(f"[DEBUG] loaded_snapshot: activities={len(data.get('activities', []))}, last_updated={data.get('last_updated')}, error={data.get('error')}")

    # 2) Auto refresh if needed
    if auto_refresh and ACTIVITY_SOURCE == "s3":
        need, reason = needs_refresh(data)
        print(f"[DEBUG] needs_refresh={need}, reason={reason}")
        if need:
            try:
                refresh_result = refresh_from_api_and_persist()
                print(f"[DEBUG] refresh_result: {json.dumps(refresh_result, ensure_ascii=False)}")
                # reload
                print(f"[DEBUG] reloading_from_s3_after_refresh: s3://{S3_BUCKET}/{S3_KEY}")
                data = s3_get_json(S3_BUCKET, S3_KEY)
                print(f"[DEBUG] reload_after_refresh: activities={len(data.get('activities', []))}, last_updated={data.get('last_updated')}")
            except Exception as e:
                print(f"[ERROR] auto refresh failed: {e}")
                data["error"] = f"auto_refresh_failed:{str(e)}"

    # 3) cache and return
    ACTIVITY_CACHE = data
    return data


# =====================
# Activity interface for LLM
# =====================
def fetch_activity_data(user_text: str) -> Dict[str, Any]:
    # auto refresh enabled
    data = load_activities(auto_refresh=True)
    all_activities = data.get("activities", [])
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

    if not OPENAI_API_KEY:
        if not activities:
            return "目前活動資料為空（S3 可能尚未更新成功）。\n你可以稍後再試或觸發更新。"
        return "（未設定 OpenAI 金鑰，先給你活動清單）\n" + "\n".join(
            f"・{a.get('title')}（{a.get('start_date')}~{a.get('end_date') or a.get('start_date')}，{a.get('location')}）"
            for a in activities[:6]
        )

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
        print("[DEBUG] calling OpenAI...")
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            resp_body = resp.read().decode("utf-8")
        resp_json = json.loads(resp_body)
        content = resp_json["choices"][0]["message"]["content"]
        print("[DEBUG] OpenAI done")
        return (content or "").strip()
    except Exception as e:
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
    # Only treat as scheduled when it looks like EventBridge
    if event.get("source") == "aws.events":
        return True
    if event.get("detail-type") in ("Scheduled Event", "EventBridge Scheduler"):
        return True
    return False


# =====================
# Lambda handler
# =====================
def lambda_handler(event, context):
    # ---- Scheduled update (EventBridge) ----
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
    path = event.get("rawPath") or event.get("path", "/")
    method = (
        event.get("requestContext", {}).get("http", {}).get("method")
        or event.get("httpMethod", "GET")
    )
    print(f"[DEBUG] method={method}, path={path}")

    # GET /
    if method == "GET" and path == "/":
        data = load_activities(auto_refresh=True)  # ✅ this is the behavior you expect
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
    if method == "GET" and path == "/refresh":
        qs = event.get("rawQueryString") or ""
        params = urllib.parse.parse_qs(qs)
        token = (params.get("token") or [""])[0]
        if REFRESH_TOKEN and token != REFRESH_TOKEN:
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
    if method == "POST":
        body = event.get("body", "") or ""
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode("utf-8")

        headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
        signature = headers.get("x-line-signature", "")

        if not verify_signature(CHANNEL_SECRET, body, signature):
            print("[ERROR] invalid signature")
            return {"statusCode": 403, "body": "Invalid signature"}

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return {"statusCode": 400, "body": "Invalid JSON"}

        events = data.get("events", [])
        for ev in events:
            if ev.get("type") != "message":
                continue
            if ev.get("message", {}).get("type") != "text":
                continue

            reply_token = ev.get("replyToken")
            user_text = ev.get("message", {}).get("text", "").strip()
            if not reply_token or not user_text:
                continue

            # ✅ Auto-refresh happens inside fetch_activity_data -> load_activities(auto_refresh=True)
            activity_json = fetch_activity_data(user_text)
            reply_text = ask_llm_about_activities(user_text, activity_json)
            reply_message(reply_token, reply_text)

        return {"statusCode": 200, "body": "OK"}

    return {"statusCode": 404, "body": "Not Found"}
