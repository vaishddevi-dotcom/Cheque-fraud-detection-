# -*- coding: utf-8 -*-
"""
PySpark + Amazon Bedrock (Amazon Nova) cheque OCR-ish understanding & field extraction
+ PhoenixDB UPSERT into Cloudera Operational Database (COD).

Improvements vs previous version:
- Fixed Bedrock payload to Nova's messages-v1 schema (no "type" keys in content parts).
- Supports Nova Premier via Inference Profile ARN (set env BEDROCK_MODEL_ID), or falls back to Nova Pro on-demand.
- Robust IFSC/account/cheque/date parsing retained.
- Fraud check after normalization to reduce false positives.
- Optional Phoenix password override via PHOENIX_PASS env var.
- Keeps amount printing non-scientific.

Notes:
- Nova Premier in us-east-1 must be invoked via an inference profile (modelId = inferenceProfileArn).
  If you don't have a profile, use Nova Pro on-demand: modelId = "amazon.nova-pro-v1:0".
- Request body uses schemaVersion="messages-v1" per Bedrock InvokeModel for Nova.
"""

from pyspark.sql import SparkSession
import base64
import json
import os
import sys
import re
import mimetypes
from datetime import datetime, date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

# ---- AWS / Bedrock ----
import boto3
from botocore.exceptions import ClientError, BotoCoreError
from botocore.config import Config

# ---- PhoenixDB / COD ----
import phoenixdb
import phoenixdb.cursor
from requests.auth import HTTPBasicAuth

# ======================= SPARK =======================
spark = SparkSession.builder.appName("InvokeBedrockNova_FilteredExtract_Upsert_Batch_Fraud_NoSci_v4").getOrCreate()

# ======================= CONFIG =======================
# Input
s3_path = "s3a://cloudera-poc-buck/cheques/incoming/"
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}

# ---- BEDROCK ----
aws_region = "us-east-1"

# Prefer inference profile ARN for Nova Premier if provided, else fall back to Nova Pro on-demand
bedrock_model_id = os.getenv("BEDROCK_MODEL_ID", "").strip() or "amazon.nova-pro-v1:0"

# ---- COD / PHOENIX (HARDCODED with env override) ----
PHOENIX_URL   = "https://cod-1iuh2nynfx25k-gateway0.hexaware.dvd1-p7f4.a0.cloudera.site/cod-1iuh2nynfx25k/cdp-proxy-api/avatica/"
PHOENIX_USER  = "csso_vaishnavidevendrad"
PHOENIX_PASS  = os.getenv("PHOENIX_PASS", "Vaish@cloudera1")   # override via env if needed
PHOENIX_VERIFY = True

# ---- Behavior flags ----
READBACK_VERIFY = True

# =================== HADOOP FILE HELPERS ===================
sc = spark.sparkContext
jvm = sc._jvm

def list_image_files(folder_uri: str):
    """
    List all files with supported image extensions under the given folder.
    Returns a sorted list of full URIs.
    """
    path = jvm.org.apache.hadoop.fs.Path(folder_uri)
    fs = path.getFileSystem(sc._jsc.hadoopConfiguration())
    if not fs.exists(path):
        return []
    files = []
    for st in fs.listStatus(path):
        if st.isFile():
            p = st.getPath().toString()
            ext = os.path.splitext(p)[1].lower()
            if ext in SUPPORTED_EXT:
                files.append(p)
    files.sort()
    return files

def read_bytes(file_uri: str) -> bytes:
    fs_path = jvm.org.apache.hadoop.fs.Path(file_uri)
    fs = fs_path.getFileSystem(sc._jsc.hadoopConfiguration())
    in_stream = fs.open(fs_path)
    try:
        # If your Hadoop client doesn't support readAllBytes(), replace with manual read loop
        data = bytearray(in_stream.readAllBytes())
    finally:
        in_stream.close()
    return bytes(data)

# =================== NORMALIZATION & UTILITIES ===================
_MONTHS = {
    'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
    'JUL':7,'AUG':8,'SEP':9,'SEPT':9,'OCT':10,'NOV':11,'DEC':12
}

def _clean_text(s): return (s or "").strip()

def normalize_ifsc(raw: str):
    """
    Robust IFSC normalizer/validator:
    - Removes spaces/hyphens/odd punctuation
    - Uppercases
    - Corrects common OCR confusions contextually
      * Head (1–4) letters: digits -> letters (0->O, 1->I, 5->S, 2->Z, 8->B)
      * Position 5 forced to '0'
      * Tail (6–11) alnum: letters likely->digits (O->0, I/L->1, Z->2, S->5, B->8, Q->0)
    - Tests sliding 11-char windows for longer tokens
    Returns canonical IFSC like SBIN0000123 or None.
    """
    if not raw:
        return None
    s = re.sub(r'[^A-Za-z0-9]', '', str(raw)).upper()
    if len(s) < 8:
        return None

    def _fix_candidate(cand: str) -> str:
        cand = cand.upper()
        if len(cand) != 11:
            return cand
        map_head = {'0':'O','1':'I','5':'S','2':'Z','8':'B'}
        head = ''.join(map_head.get(ch, ch) for ch in cand[:4])
        mid = '0'  # enforce zero at pos5
        map_tail = {'O':'0','I':'1','L':'1','Z':'2','S':'5','B':'8','Q':'0'}
        tail = ''.join(map_tail.get(ch, ch) for ch in cand[5:])
        return head + mid + tail

    def _is_valid_ifsc(x: str) -> bool:
        return bool(re.match(r'^[A-Z]{4}0[A-Z0-9]{6}$', x))

    if len(s) == 11:
        cand = _fix_candidate(s)
        if _is_valid_ifsc(cand):
            return cand

    if len(s) > 11:
        for i in range(0, len(s) - 11 + 1):
            window = _fix_candidate(s[i:i+11])
            if _is_valid_ifsc(window):
                return window

    return None

def try_parse_date_yyyy_mm_dd(raw: str):
    """
    Try many date patterns and return YYYY-MM-DD or None.
    Handles 'Date:' prefix, ordinal suffixes, commas.
    """

    if not raw:
        return None

    s = _clean_text(raw)
    s = re.sub(r'(?i)\bdate\s*[:\-]?\s*', '', s)
    s = s.replace(',', ' ').replace('  ', ' ')
    s = re.sub(r'(\d{1,2})(st|nd|rd|th)\b', r'\1', s, flags=re.IGNORECASE)

    # -------- HANDLE 15092026 / 150926 --------
    if re.match(r'^\d{8}$', s):
        try:
            dt = datetime.strptime(s, "%d%m%Y").date()
            return dt.strftime("%Y-%m-%d")
        except:
            pass

    if re.match(r'^\d{6}$', s):
        try:
            dt = datetime.strptime(s, "%d%m%y").date()
            return dt.strftime("%Y-%m-%d")
        except:
            pass
    # ------------------------------------------

    numeric_patterns = [
        "%d/%m/%Y","%d-%m-%Y","%d.%m.%Y",
        "%d/%m/%y","%d-%m-%y","%d.%m.%y",
        "%m/%d/%Y","%m-%d-%Y","%m.%d.%Y",
        "%m/%d/%y","%m-%d-%y","%m.%d.%y"
    ]

    for fmt in numeric_patterns:
        try:
            dt = datetime.strptime(s, fmt).date()
            return dt.strftime("%Y-%m-%d")
        except:
            pass

    m = re.match(r'(?i)^\s*(\d{1,2})[\s\-\.]+([A-Za-z]{3,})[\s\-\.]+(\d{2,4})\s*$', s)
    if m:
        d, mon, y = m.groups()
        mon = mon[:3].upper()
        if mon in _MONTHS:
            yy = int(y) + (2000 if len(y) == 2 else 0) if len(y) == 2 else int(y)
            try:
                return date(yy, _MONTHS[mon], int(d)).strftime("%Y-%m-%d")
            except:
                pass

    m2 = re.match(r'(?i)^\s*([A-Za-z]{3,})[\s\-\.]+(\d{1,2})[\s\-\,\.]+(\d{2,4})\s*$', s)
    if m2:
        mon, d, y = m2.groups()
        mon = mon[:3].upper()
        if mon in _MONTHS:
            yy = int(y) + (2000 if len(y) == 2 else 0) if len(y) == 2 else int(y)
            try:
                return date(yy, _MONTHS[mon], int(d)).strftime("%Y-%m-%d")
            except:
                pass

    return None

def _extract_first(regex, text, flags=0):
    m = re.search(regex, text, flags)
    return m.group(1) if m else None

def fmt_money(val):
    if val is None:
        return None
    if not isinstance(val, Decimal):
        try:
            val = Decimal(str(val))
        except Exception:
            return str(val)
    val = val.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{val:.2f}"

# =================== CHEQUE NUMBER EXTRACTOR ===================
def extract_cheque_number(lines: list, account_number: str = None, ifsc_code: str = None):
    """
    Heuristics to find a cheque number:
    - Prefer explicit labels: 'Cheque No', 'Chq No', 'CHQ No', 'CQ No', 'Leaf No', 'Slip No', 'No:'
    - Else pick a 6–10 digit run not equal/substring of account number, not an amount, not a date, not IFSC
    - Prefer longer and earlier candidates
    Returns a string or None.
    """
    acct_digits = re.sub(r'\D', '', account_number or "")
    label_patterns = [
        r'(?i)\b(?:che?que|chq|cq|leaf|slip)\s*(?:no|number|#)?\s*[:\-]?\s*([0-9 \-]{6,12})',
        r'(?i)\bNo\.?\s*[:\-]?\s*([0-9 \-]{6,12})'
    ]
    def _clean_digits(x): return re.sub(r'\D', '', x or '')

    # 1) Label-based
    for l in lines or []:
        for pat in label_patterns:
            m = re.search(pat, l)
            if m:
                cand = _clean_digits(m.group(1))
                if 6 <= len(cand) <= 10:
                    if acct_digits and (cand == acct_digits or cand in acct_digits or acct_digits in cand):
                        continue
                    return cand

    # 2) Unlabeled candidates
    candidates = []
    for l in lines or []:
        U = (l or "").upper()
        if re.search(r'(₹|INR|\$|RUPEES|DOLLARS)', U): 
            continue
        if re.search(r'\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4}', l): 
            continue
        if 'IFSC' in U or 'IFS CODE' in U:
            continue
        for run in re.findall(r'(\d{6,10})', l):
            if acct_digits and (run == acct_digits or run in acct_digits or acct_digits in run):
                continue
            candidates.append(run)

    if candidates:
        # Prefer longer length first, then keep earliest by natural collection order
        candidates.sort(key=lambda x: (-len(x),))
        return candidates[0]

    return None

# =================== FALLBACK PARSER (robust) ===================
def parse_cheque_fields(lines: list, image_name: str):
    """
    Robust regex-based extraction from list of lines.
    """
    # Dedup while preserving order
    uniq_lines, seen = [], set()
    for l in lines or []:
        if l not in seen:
            uniq_lines.append(l)
            seen.add(l)
    lines = uniq_lines
    text = "\n".join(lines)

    # Initialize
    cheque_number = None
    cheque_date = None
    payee_name = None
    amount_numbers = None
    amount_words = None
    bank_name = None
    account_number = None
    ifsc_code = None

    # IFSC extraction first (labels + label-free using robust normalizer)
    for l in lines:
        m = re.search(r'(?i)\bIFSC(?:\s*Code)?\b[:\-]?\s*([A-Za-z0-9 \-]{5,})', l)
        if not m:
            m = re.search(r'(?i)\bIFS\s*Code\b[:\-]?\s*([A-Za-z0-9 \-]{5,})', l)
        if m:
            cand = normalize_ifsc(m.group(1))
            if cand:
                ifsc_code = cand
                break
    if not ifsc_code:
        for l in lines:
            for tok in re.findall(r'[A-Za-z0-9][A-Za-z0-9 \-]{9,}', l):
                cand = normalize_ifsc(tok)
                if cand:
                    ifsc_code = cand
                    break
            if ifsc_code:
                break

    # Account number extraction (labels first, then longest 9–18 digits)
    acct_patterns = [
        r'(?i)\b(?:account|a\/c|ac|acc)\s*(?:no|number|#)?\s*[:\-]?\s*([0-9 \-]{6,})',
        r'(?i)\bA\/C\s*[:\-#]?\s*([0-9 \-]{6,})',
        r'(?i)\bAcc(?:ount)?\s*No\.?\s*[:\-]?\s*([0-9 \-]{6,})'
    ]
    def _normalize_acct(x):
        if not x: return None
        d = re.sub(r'\D', '', x)
        if 6 <= len(d) <= 20:
            return d
        return None

    for l in lines:
        for pat in acct_patterns:
            m = re.search(pat, l)
            if m:
                account_number = _normalize_acct(m.group(1))
                if account_number:
                    break
        if account_number:
            break
    if not account_number:
        candidates = []
        for l in lines:
            U = (l or "").upper()
            if re.search(r'(₹|INR|\$|RUPEES|DOLLARS)', U):
                continue
            if re.search(r'\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4}', l):
                continue
            if 'IFSC' in U:
                continue
            runs = re.findall(r'(\d{9,18})', l)
            candidates.extend(runs)
        if candidates:
            account_number = max(candidates, key=len)

    # Cheque number using context (avoid acct/ifsc)
    cheque_number = extract_cheque_number(lines, account_number=account_number, ifsc_code=ifsc_code)

    # Date (line-by-line with robust parser)
    for l in lines:
        cand = try_parse_date_yyyy_mm_dd(l)
        if cand:
            cheque_date = cand
            break
        token = _extract_first(r'(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4})', l)
        if token:
            cand = try_parse_date_yyyy_mm_dd(token)
            if cand:
                cheque_date = cand
                break

    # Payee name: line after 'PAY TO (THE ORDER OF)?'
    for idx, l in enumerate(lines):
        if re.search(r'(?i)pay\s+to(\s+the\s+order\s+of)?', l):
            if idx + 1 < len(lines):
                payee_name = lines[idx + 1].strip()
            break

    # Amount in numbers (prefer currency symbol), else large number
    for l in lines:
        m1 = re.search(r'(?:₹|Rs\.?|INR|\$)\s*([0-9][\d,]*(?:\.\d{1,2})?)', l, re.IGNORECASE)
        if m1:
            amount_numbers = m1.group(1).replace(",", "")
            break

    # IFSC
    for l in lines:
        m = re.search(r'(?i)\bIFSC(?:\s*Code)?\b[:\-]?\s*([A-Za-z0-9]{11})', l)
        if m:
            cand = normalize_ifsc(m.group(1))
            if cand:
                ifsc_code = cand
                break

    # Amount in words
    for l in lines:
        if re.search(r'(?i)\b(rupees|dollars|only)\b', l):
            amount_words = l.strip()
            break

    # Bank name: first 'BANK' not an IFSC line
    for l in lines:
        if 'BANK' in (l or '').upper() and 'IFSC' not in (l or '').upper():
            bank_name = l.strip()
            break

    return {
        "cheque_number": cheque_number,
        "cheque_date": cheque_date,
        "payee_name": payee_name,
        "amount_numbers": amount_numbers,
        "amount_words": amount_words,
        "bank_name": bank_name,
        "account_number": account_number,
        "ifsc_code": ifsc_code,
        "image_name": image_name,
        "image_uploaded_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# =================== BEDROCK INVOCATION (NOVA, messages-v1) ===================
def _mime_to_nova_format(mime: str) -> str:
    """
    Map MIME to Nova's expected 'format' token.
    """
    m = (mime or "").lower()
    if "jpeg" in m or m.endswith("/jpg") or "jpg" in m:
        return "jpeg"
    if "png" in m:
        return "png"
    if "webp" in m:
        return "webp"
    if "tif" in m:
        return "tiff"
    if "bmp" in m:
        return "bmp"
    if "gif" in m:
        return "gif"
    return "jpeg"

def invoke_bedrock_ocr_and_extract(image_bytes: bytes, mime: str, model_id: str, region: str) -> dict:
    """
    Invokes Amazon Nova (Premier via inference profile ARN or Pro on-demand) using InvokeModel
    and expects STRICT JSON in the assistant's text. Uses messages-v1 (Nova schema).
    """
    bedrock = boto3.client(
        service_name="bedrock-runtime",
        region_name=region,
        config=Config(read_timeout=3600, connect_timeout=3600, retries={'max_attempts': 3})
    )

    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    img_format = _mime_to_nova_format(mime)

    # System + user content (NO "type" fields; Nova requires { "text": ... } / { "image": {...} })
    system_list = [
        { "text": "You are a document understanding assistant for bank cheques." }
    ]

    user_content = [
        {
            "image": {
                "format": img_format,
                "source": { "bytes": img_b64 }
            }
        },
        {
            "text": (
                "TASKS:\n"
                "1) Transcribe readable text lines in natural reading order as an array.\n"
                "2) Extract the following fields (use null if unknown):\n"
                "   - cheque_number\n"
                "   - cheque_date (prefer YYYY-MM-DD; else original string)\n"
                "   - payee_name\n"
                "   - amount_numbers (numeric like 30500.00)\n"
                "   - amount_words\n"
                "   - bank_name\n"
                "   - account_number\n"
                "   - ifsc_code\n\n"
                "RESPONSE FORMAT (STRICT JSON, no commentary, no code fences):\n"
                "{\n"
                '  \"lines\": [\"...\"],\n'
                '  \"fields\": {\n'
                '    \"cheque_number\": null,\n'
                '    \"cheque_date\": null,\n'
                '    \"payee_name\": null,\n'
                '    \"amount_numbers\": null,\n'
                '    \"amount_words\": null,\n'
                '    \"bank_name\": null,\n'
                '    \"account_number\": null,\n'
                '    \"ifsc_code\": null\n'
                "  }\n"
                "}\n"
                "Do not include any additional keys. Use null for unknown values."
            )
        }
    ]

    body = json.dumps({
        "schemaVersion": "messages-v1",
        "system": system_list,                           # List of { "text": ... }
        "messages": [ { "role": "user", "content": user_content } ],
        "inferenceConfig": {
            "maxTokens": 2000,
            "temperature": 0.0,
            "topP": 0.9
        }
        # "additionalModelRequestFields": {"topK": 20}  # optional
    })

    try:
        resp = bedrock.invoke_model(
            modelId=model_id,            # Can be a base model ID (e.g., nova-pro) or an inference profile ARN (nova-premier)
            body=body,
            accept="application/json",
            contentType="application/json"
        )
        payload = json.loads(resp["body"].read().decode("utf-8"))

        # Nova returns: {"output":{"message":{"content":[{"text":"{...json...}"}]}}}
        text_out = ""
        try:
            parts = payload["output"]["message"]["content"]
            for part in parts:
                if isinstance(part, dict) and "text" in part:
                    text_out += part["text"]
        except Exception:
            pass

        cleaned = (text_out or "").strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            first_brace = cleaned.find("{")
            last_brace = cleaned.rfind("}")
            if first_brace != -1 and last_brace != -1:
                cleaned = cleaned[first_brace:last_brace+1]

        return json.loads(cleaned)

    except ClientError as e:
        raise RuntimeError(f"AWS ClientError invoking Bedrock model '{model_id}': {e}")
    except (BotoCoreError, json.JSONDecodeError, KeyError, TypeError) as e:
        raise RuntimeError(f"Bedrock invocation/parse error: {e}")

# =================== PHOENIX / COD HELPERS ===================
def _to_sql_date(cheque_date_val):
    if not cheque_date_val:
        return None
    if isinstance(cheque_date_val, date) and not isinstance(cheque_date_val, datetime):
        return cheque_date_val
    if isinstance(cheque_date_val, datetime):
        return cheque_date_val.date()
    s = str(cheque_date_val).strip()
    best = try_parse_date_yyyy_mm_dd(s)
    if best:
        try:
            return datetime.strptime(best, "%Y-%m-%d").date()
        except Exception:
            pass
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y", "%d/%m/%y", "%m/%d/%y", "%d.%m.%Y", "%d.%m.%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    return None

def _to_sql_decimal(amount_numbers_val):
    if amount_numbers_val is None:
        return None
    s = str(amount_numbers_val).strip().replace(",", "")
    s = re.sub(r"[^\d.\-]", "", s)
    if s == "":
        return None
    try:
        d = Decimal(s)
        d = d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return d
    except (InvalidOperation, ValueError):
        return None

def _to_sql_timestamp(ts_val):
    if ts_val is None:
        return None
    if isinstance(ts_val, datetime):
        return ts_val
    s = str(ts_val).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return datetime.now()

def phoenix_connect():
    return phoenixdb.connect(
        url=PHOENIX_URL,
        autocommit=True,
        auth=HTTPBasicAuth(PHOENIX_USER, PHOENIX_PASS),
        serialization="PROTOBUF",
        verify=PHOENIX_VERIFY
    )

def upsert_into_cod(conn, fields: dict):
    cheque_number = fields.get("cheque_number")
    if not cheque_number or str(cheque_number).strip() == "":
        raise RuntimeError("CHEQUE_OCR UPSERT aborted: 'cheque_number' is required (PRIMARY KEY).")

    sql_date = _to_sql_date(fields.get("cheque_date"))
    sql_amount = _to_sql_decimal(fields.get("amount_numbers"))
    sql_ts = _to_sql_timestamp(fields.get("image_uploaded_time"))

    values = (
        str(cheque_number).strip(),
        sql_date,
        (fields.get("payee_name") or None),
        sql_amount,
        (fields.get("amount_words") or None),
        (fields.get("bank_name") or None),
        (fields.get("account_number") or None),
        (fields.get("ifsc_code") or None),
        (fields.get("image_name") or None),
        sql_ts
    )

    sql = (
        'UPSERT INTO "CHEQUE_OCR"('
        '"Cheque number","Date on Cheque","Payee name","Amount in Numbers","Amount in words",'
        '"bank name","Account number","IFSC code","image name","image uploaded time"'
        ") VALUES (?,?,?,?,?,?,?,?,?,?)"
    )

    cur = None
    try:
        cur = conn.cursor()
        cur.execute(sql, values)
        print(f"[COD] UPSERT OK for Cheque number = {cheque_number}")
    finally:
        try:
            if cur: cur.close()
        except Exception:
            pass

def readback_from_cod(conn, cheque_number: str):
    sql = (
        'SELECT "Cheque number","Date on Cheque","Payee name",'
        'CAST("Amount in Numbers" AS DECIMAL(18,2)) AS "Amount in Numbers",'
        '"Amount in words","bank name","Account number","IFSC code","image name","image uploaded time" '
        'FROM "CHEQUE_OCR" WHERE "Cheque number" = ?'
    )
    cur = None
    try:
        cur = conn.cursor()
        cur.execute(sql, (cheque_number,))
        row = cur.fetchone()
        if not row:
            print(f"[COD] No row found for Cheque number = {cheque_number}")
            return
        cols = [
            "Cheque number","Date on Cheque","Payee name","Amount in Numbers",
            "Amount in words","bank name","Account number","IFSC code",
            "image name","image uploaded time"
        ]
        print("\n[COD] Read-back row:")
        for c, v in zip(cols, row):
            if c == "Amount in Numbers":
                print(f"  {c}: {fmt_money(v)}")
            else:
                print(f"  {c}: {v}")
        print()
    finally:
        try:
            if cur: cur.close()
        except Exception:
            pass

# =================== FRAUD DETECTION ===================
def detect_fraud(fields: dict):
    """
    Returns (is_fraud: bool, missing_columns: list[str]).
    Checks after normalization.
    """
    missing = []

    def _is_blank(v):
        return v is None or (isinstance(v, str) and v.strip() == "")

    # Primary key
    if _is_blank(fields.get("cheque_number")):
        missing.append("Cheque number")

    # Date
    if _to_sql_date(fields.get("cheque_date")) is None:
        missing.append("Date on Cheque")

    # Strings / numeric
    if _is_blank(fields.get("payee_name")):
        missing.append("Payee name")
    if _to_sql_decimal(fields.get("amount_numbers")) is None:
        missing.append("Amount in Numbers")
    if _is_blank(fields.get("amount_words")):
        missing.append("Amount in words")
    if _is_blank(fields.get("bank_name")):
        missing.append("bank name")

    # Account number normalize
    acct = fields.get("account_number")
    acct_digits = re.sub(r'\D', '', acct or "")
    if len(acct_digits) < 6 or len(acct_digits) > 20:
        missing.append("Account number")
    else:
        fields["account_number"] = acct_digits  # normalize in-place

    # IFSC normalize
    ifsc_norm = normalize_ifsc(fields.get("ifsc_code"))
    if not ifsc_norm:
        missing.append("IFSC code")
    else:
        fields["ifsc_code"] = ifsc_norm

    if _is_blank(fields.get("image_name")):
        missing.append("image name")

    ts = _to_sql_timestamp(fields.get("image_uploaded_time"))
    if ts is None:
        missing.append("image uploaded time")

    return (len(missing) > 0), missing

# =================== MERGE LOGIC ===================
def merge_fields(model_fields: dict, fallback_fields: dict):
    """
    Per-field merge: prefer non-empty model value, otherwise fallback.
    Always set image_name & timestamp from runtime/fallback.
    """
    out = {}
    keys = ["cheque_number","cheque_date","payee_name","amount_numbers","amount_words",
            "bank_name","account_number","ifsc_code"]
    for k in keys:
        mv = model_fields.get(k) if isinstance(model_fields, dict) else None
        if isinstance(mv, str) and mv.strip() == "":
            mv = None
        out[k] = mv if mv is not None else fallback_fields.get(k)

    out["image_name"] = fallback_fields.get("image_name")
    out["image_uploaded_time"] = fallback_fields.get("image_uploaded_time")
    return out

# =================== MAIN (BATCH) ===================
try:
    image_files = list_image_files(s3_path)
    if not image_files:
        print(f"No images found under {s3_path} with extensions: {sorted(SUPPORTED_EXT)}")
        spark.stop()
        sys.exit(0)

    # Log model selection
    if bedrock_model_id.startswith("arn:aws:bedrock:") and ":inference-profile/" in bedrock_model_id:
        print(f"[Bedrock] Using Nova Premier via Inference Profile: {bedrock_model_id}")
    else:
        print(f"[Bedrock] Using on-demand model: {bedrock_model_id} (recommended fallback: amazon.nova-pro-v1:0)")

    print(f"Found {len(image_files)} image(s) under {s3_path}")
    cod_conn = phoenix_connect()

    processed = 0
    upserts_ok = 0
    frauds = []
    failures = []

    for image_file in image_files:
        processed += 1
        print("\n" + "="*90)
        print(f"[{processed}/{len(image_files)}] Processing: {image_file}")
        print("="*90)
        try:
            guessed_mime, _ = mimetypes.guess_type(image_file)
            mime = guessed_mime or "image/jpeg"

            img_bytes = read_bytes(image_file)

            # 1) Bedrock extraction (Nova)
            bedrock_out = invoke_bedrock_ocr_and_extract(
                image_bytes=img_bytes,
                mime=mime,
                model_id=bedrock_model_id,
                region=aws_region
            )

            # 2) Collect raw lines (dedup, preserve order)
            text_lines = []
            if isinstance(bedrock_out, dict):
                raw_lines = bedrock_out.get("lines", [])
                if isinstance(raw_lines, list):
                    seen = set()
                    for t in raw_lines:
                        if t not in seen:
                            text_lines.append(t)
                            seen.add(t)

            # 3) Model fields if present
            model_fields = {}
            if isinstance(bedrock_out, dict) and isinstance(bedrock_out.get("fields"), dict):
                model_fields = bedrock_out["fields"]

            # 4) Fallback parse from lines
            fallback = parse_cheque_fields(text_lines, image_name=os.path.basename(image_file))

            # 5) Merge field-by-field
            fields = merge_fields(model_fields, fallback)

            # ---- Log Output ----
            print("\n---- Extracted Raw Lines (from Bedrock) ----")
            for t in text_lines:
                print(t)

            # pretty copy for logs (non-scientific money)
            log_fields = dict(fields)
            def _safe_money(v):
                d = _to_sql_decimal(v)
                return fmt_money(d) if d is not None else None
            log_fields["amount_numbers"] = _safe_money(fields.get("amount_numbers"))
            print("\n---- MERGED STRUCTURED OUTPUT (pre-check) ----")
            print(json.dumps(log_fields, indent=2, default=str))

            # 6) FRAUD CHECK (after normalization)
            is_fraud, missing_cols = detect_fraud(fields)
            if is_fraud:
                print(f"\n[FRAUD] {fields.get('image_name', os.path.basename(image_file))} is fraud – missing/null: {', '.join(missing_cols)}\n")
                frauds.append(f"{os.path.basename(image_file)} :: {', '.join(missing_cols)}")
                continue

            # 7) UPSERT (only non-fraud)
            upsert_into_cod(cod_conn, fields)
            upserts_ok += 1

            # 8) Read-back verification
            if READBACK_VERIFY and fields.get("cheque_number"):
                readback_from_cod(cod_conn, fields["cheque_number"])

        except Exception as e:
            msg = f"{os.path.basename(image_file)}: {repr(e)}"
            print(f"[ERROR] {msg}", file=sys.stderr)
            failures.append(msg)

    # Close Phoenix connection
    try:
        cod_conn.close()
    except Exception:
        pass

    # ================= SUMMARY =================
    print("\n" + "#"*90)
    print("BATCH SUMMARY")
    print("#"*90)
    print(f"Total images found   : {len(image_files)}")
    print(f"Processed            : {processed}")
    print(f"UPSERT success       : {upserts_ok}")
    print(f"Fraud detected       : {len(frauds)}")
    print(f"Failures             : {len(failures)}")
    if frauds:
        print("\nFraud details (image :: missing columns):")
        for f in frauds:
            print(f" - {f}")
    if failures:
        print("\nFailure details:")
        for f in failures:
            print(f" - {f}")
    print("#"*90 + "\n")

except Exception as e:
    print(f"Fatal Error: {e}", file=sys.stderr)
    raise
finally:
    spark.stop()

