# -*- coding: utf-8 -*-
# PySpark + Bedrock (Nova) cheque OCR pipeline -> Cloudera COD (PhoenixDB)

import os, re, sys, json, mimetypes
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import boto3
from botocore.exceptions import ClientError, BotoCoreError
import phoenixdb
from requests.auth import HTTPBasicAuth
from pyspark.sql import SparkSession

# CONFIG
s3_path          = "s3a://cloudera-poc-buck/cheques/incoming/"
aws_region       = "us-east-1"
bedrock_model_id = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0")
PHOENIX_URL      = ("https://cod-1iuh2nynfx25k-gateway0.hexaware.dvd1-p7f4.a0.cloudera.site"
                    "/cod-1iuh2nynfx25k/cdp-proxy-api/avatica/")
PHOENIX_USER     = "csso_vaishnavidevendrad"
PHOENIX_PASS     = os.environ.get("PHOENIX_PASS", "Vaish@cloudera1")
PHOENIX_VERIFY   = False
RECREATE_TABLE   = True   # DROP+recreate Cheque_data each run; set False after first run
SUPPORTED_EXT    = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}

# SPARK
spark = SparkSession.builder.appName("ChequeOCR").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
sc  = spark.sparkContext
jvm = sc._jvm

# S3 HELPERS
def list_image_files(folder_uri):
    path = jvm.org.apache.hadoop.fs.Path(folder_uri)
    fs   = path.getFileSystem(sc._jsc.hadoopConfiguration())
    if not fs.exists(path): return []
    return sorted(
        st.getPath().toString() for st in fs.listStatus(path)
        if st.isFile() and
        os.path.splitext(st.getPath().toString())[1].lower() in SUPPORTED_EXT
    )

def read_bytes(uri):
    p = jvm.org.apache.hadoop.fs.Path(uri)
    fs = p.getFileSystem(sc._jsc.hadoopConfiguration())
    s  = fs.open(p)
    try:    return bytes(bytearray(s.readAllBytes()))
    finally: s.close()

# NORMALIZATION HELPERS
_ADDRESS_KW = re.compile(
    r'(?i)\b(road|rd|street|st|nagar|marg|mall|lane|avenue|colony|sector|block|'
    r'plot|flat|floor|building|complex|bridge|chowk|bazaar|market|plaza|tower|'
    r'park|garden|village|near|opp|opposite|india|mumbai|delhi|bengaluru|kolkata|'
    r'chennai|hyderabad|pune|ahmedabad|jaipur|lucknow)\b'
)

def _is_address_line(line):
    return bool(',' in line and _ADDRESS_KW.search(line))

def _is_micr_line(line):
    return bool(re.search(r'\d{6,}\s*:', line) or
                re.search(r'[\u0900-\u097F].*\d{5,}|"\s*\d{5,}\s*"', line))

def _is_pincode(token, full_line=""):
    d = re.sub(r'\D', '', token)
    if len(d) != 6 or not re.match(r'^[1-8]\d{5}$', d): return False
    return bool(re.search(r'(?i)(india|pin|postal|-\s*' + d + r')', full_line))

_BARE8 = re.compile(r'^\d{8}$')
_MONTHS = {'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
           'JUL':7,'AUG':8,'SEP':9,'SEPT':9,'OCT':10,'NOV':11,'DEC':12}

def _try_ddmmyyyy(s):
    if not _BARE8.match(s): return None
    dd, mm, yyyy = int(s[0:2]), int(s[2:4]), int(s[4:8])
    try:
        if 1 <= dd <= 31 and 1 <= mm <= 12 and 1900 <= yyyy <= 2100:
            return date(yyyy, mm, dd).strftime("%Y-%m-%d")
    except ValueError: pass
    return None

def try_parse_date(raw):
    if not raw: return None
    s = str(raw).strip()
    # Try DDMMYYYY from bare digits
    r = _try_ddmmyyyy(re.sub(r'\D', '', s))
    if r: return r
    # Try with separators (FIXED: proper capture groups)
    m = re.match(r'(\d{1,2})[\/\.\-](\d{1,2})[\/\.\-](\d{2,4})', s)
    if m:
        dd, mm, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        yyyy = yy + 2000 if yy < 100 else yy
        try:
            if 1 <= dd <= 31 and 1 <= mm <= 12:
                return date(yyyy, mm, dd).strftime("%Y-%m-%d")
        except ValueError: pass
    # Try "15 SEP 2026"
    m = re.search(r'(\d{1,2})\s+([A-Za-z]{3,4})\s+(\d{4})', s)
    if m:
        dd = int(m.group(1)); mon = m.group(2).upper()[:4]; yyyy = int(m.group(3))
        if mon in _MONTHS:
            try: return date(yyyy, _MONTHS[mon], dd).strftime("%Y-%m-%d")
            except ValueError: pass
    # Try ISO-ish
    m = re.match(r'(\d{4})-(\d{2})-(\d{2})', s)
    if m:
        try: return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).strftime("%Y-%m-%d")
        except ValueError: pass
    return None

def fmt_money(val):
    if val is None: return None
    try: return str(Decimal(str(val).replace(',','')).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
    except Exception: return str(val)

# IFSC NORMALIZER

def normalize_ifsc(raw):
    if not raw: return None
    s = re.sub(r'[^A-Z0-9]', '', str(raw).upper())
    if len(s) < 11: return None
    # Bank code (first 4 chars): only fix digits misread as letters (0->O, 1->I etc)
    # Do NOT map letters->digits here — that corrupts valid codes like HDFC (D is not 0)
    _D2L = {'0':'O','1':'I','5':'S','8':'B'}
    head = ''.join(_D2L.get(c, c) for c in s[:4])
    if not head.isalpha(): return None
    if s[4] not in ('0','O'): return None
    result = head + '0' + s[5:11]
    return result if re.match(r'^[A-Z]{4}0[A-Z0-9]{6}$', result) else None


def extract_ifsc_from_lines(lines):
    label_re = re.compile(
        r'(?i)(?:RTGS\s*/\s*(?:NEFT\s+)?|(?:NEFT\s+)?)IFSC\s*[:/]?\s*([A-Z0-9]{8,15})?'
    )
    found_label = False; label_was_empty = False
    for line in lines:
        m = label_re.search(line)
        if m:
            found_label = True
            raw_val = (m.group(1) or "").strip()
            if not raw_val:
                label_was_empty = True; continue
            c = normalize_ifsc(raw_val)
            if c: return c
    if label_was_empty: return None
    for line in lines:
        if _is_address_line(line): continue
        for tok in re.findall(r'[A-Z0-9]{8,15}', line.upper()):
            c = normalize_ifsc(tok)
            if c: return c
    return None

# CHEQUE NUMBER EXTRACTOR (MICR-first)
_MICR_RE = re.compile(
    r'["\u201C\u201D][\u0900-\u097F\s*]*(\d{5,8})[\u0900-\u097F\s*]*["\u201C\u201D*]'
)

def extract_cheque_number(lines, account_number=None):
    acct = re.sub(r'\D', '', account_number or "")
    # Pass 1: MICR band
    for l in lines or []:
        if _is_micr_line(l):
            m = _MICR_RE.search(l)
            if m:
                cand = m.group(1)
                if acct and cand == acct: continue
                if acct and len(cand) >= 9 and (cand in acct or acct in cand): continue
                return cand
    # Pass 2: explicit label
    for l in lines or []:
        m = re.search(r'(?i)\b(?:che?que|chq|leaf)\s*(?:no|#)?\s*[:\-]?\s*([0-9 \-]{6,12})', l)
        if m:
            cand = re.sub(r'\D', '', m.group(1))
            if 6 <= len(cand) <= 10 and cand != acct: return cand
    # Pass 3: conservative 6-digit unlabeled scan
    for l in lines or []:
        U = l.upper()
        if re.search(r'[RS]|RUPEES|IFSC', U) and '\u20b9' not in l and '$' not in l:
            pass
        if re.search(r'(?i)(ifsc|rupees|soit)', l): continue
        if _is_address_line(l) or _is_micr_line(l): continue
        for run in re.findall(r'(\d{6})', l):
            if acct and (run in acct or acct in run): continue
            if _try_ddmmyyyy(run): continue
            return run
    return None

# FALLBACK PARSER
_BANK_RE = re.compile(
    r'(?i)\b(HDFC\s*BANK|ICICI\s*BANK|STATE\s*BANK\s*OF\s*INDIA|SBI|'
    r'BANK\s*OF\s*BARODA|BOB|AXIS\s*BANK|KOTAK|YES\s*BANK|PNB|'
    r'PUNJAB\s*NATIONAL|CANARA|UNION\s*BANK|IDBI|FEDERAL\s*BANK)\b'
)

def _normalize_acct(raw):
    d = re.sub(r'\D', '', raw or '')
    return d if 6 <= len(d) <= 20 else None

def parse_cheque_fields(lines, image_name=""):
    f = {k: None for k in ['cheque_number','cheque_date','payee_name',
         'amount_numbers','amount_words','bank_name','account_number',
         'ifsc_code','image_name','image_uploaded_time','amount_number_raw_line']}
    f['image_name']          = image_name
    f['image_uploaded_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # bank name
    for l in lines:
        m = _BANK_RE.search(l)
        if m: f['bank_name'] = m.group(1).strip(); break

    # date
    consumed_date = set()
    for l in lines:
        d = re.sub(r'\D', '', l)
        r = _try_ddmmyyyy(d)
        if r: f['cheque_date'] = r; consumed_date.add(d); break
    if not f['cheque_date']:
        for l in lines:
            r = try_parse_date(l)
            if r: f['cheque_date'] = r; consumed_date.add(re.sub(r'\D','',r)); break

    # payee
    for i, l in enumerate(lines):
        if re.search(r'(?i)\bpay\b', l):
            rest = re.sub(r'(?i)\bpay\b', '', l).strip().strip(':').strip()
            if len(rest) > 2 and not re.search(r'(?i)(bearer|order)', rest):
                f['payee_name'] = rest
            elif i+1 < len(lines):
                nxt = lines[i+1].strip()
                if nxt and not re.search(r'(?i)(or bearer|rupees|soit)', nxt):
                    f['payee_name'] = nxt
            if f['payee_name']: break

    # amount words
    for i, l in enumerate(lines):
        if re.search(r'(?i)\brupees\b', l):
            rest = re.sub(r'(?i)\brupees\b|[\u0900-\u097F]+', '', l).strip()
            if len(rest) > 5: f['amount_words'] = rest
            elif i+1 < len(lines):
                nxt = re.sub(r'[\u0900-\u097F]+', '', lines[i+1]).strip()
                if re.search(r'(?i)\b(thousand|lakh|crore|hundred|only)\b', nxt):
                    f['amount_words'] = nxt
            if f['amount_words']: break

    # amount numbers — SOIT box
    skip_re = re.compile(r'(?i)(ifsc|rtgs|neft|branch|br:|gm ro|a\/c|min def)')
    for l in lines:
        if _is_address_line(l) or _is_micr_line(l) or skip_re.search(l): continue
        m = re.search(r'[₹\$]\s*[\d,]+(?:\.\d{1,2})?(?:\s*/[-\u2013]?)?', l)
        if m:
            raw = re.sub(r'[^\d.]', '', m.group(0))
            d = re.sub(r'\D', '', raw)
            if d and d not in consumed_date and not _is_pincode(d, l):
                f['amount_numbers'] = fmt_money(raw); f['amount_number_raw_line'] = m.group(0); break
    if not f['amount_numbers']:
        for l in lines:
            if _is_address_line(l) or _is_micr_line(l) or skip_re.search(l): continue
            m = re.search(r'([\d,]+(?:\.\d{1,2})?)\s*/[-]?', l)
            if m:
                raw = m.group(1).replace(',', '')
                d = re.sub(r'\D', '', raw)
                if d not in consumed_date and not _is_pincode(d, l) and len(d) > 3:
                    f['amount_numbers'] = fmt_money(raw); f['amount_number_raw_line'] = m.group(0); break

    # account number
    for l in lines:
        if re.search(r'(?i)(gm ro|min def|minde)', l):
            for tok in l.split():
                a = _normalize_acct(tok)
                if a: f['account_number'] = a; break
        if f['account_number']: break
    if not f['account_number']:
        for l in lines:
            m = re.search(r'\b(\d{9,18})\b', l)
            if m:
                a = _normalize_acct(m.group(1))
                if a: f['account_number'] = a; break

    f['cheque_number'] = extract_cheque_number(lines, f.get('account_number'))
    f['ifsc_code']     = extract_ifsc_from_lines(lines)
    return f

# BEDROCK INVOCATION
_BEDROCK = boto3.client("bedrock-runtime", region_name=aws_region)

def _mime_to_nova(mime):
    return {"image/jpeg":"jpeg","image/png":"png","image/gif":"gif",
            "image/webp":"webp","image/tiff":"tiff","image/bmp":"bmp"}.get(mime,"jpeg")

def invoke_bedrock(image_bytes, mime, model_id):
    import base64
    prompt = (
        "You are a cheque OCR expert. Extract fields from this cheque image.\n"
        "- cheque_number: MICR band (bottom line), first quoted digit group e.g. \"098765\". NOT the date.\n"
        "- cheque_date: from D D M M Y Y Y Y box (top-right). Return raw e.g. '15092026'. 'DDMMYYYY' is a label not a date.\n"
        "- payee_name: name after 'Pay'\n"
        "- amount_numbers: numeric from SOIT box ONLY. Null if SOIT box is empty.\n"
        "- amount_words: text after 'Rupees'\n"
        "- bank_name: bank from logo/header\n"
        "- account_number: from Gm Ro / min def box\n"
        "- ifsc_code: 11-char code after RTGS/NEFT IFSC. Null if absent.\n\n"
        "Return ONLY JSON: {\"lines\": [...all text lines...], \"fields\": {...}}. No markdown."
    )
    body = json.dumps({
        "schemaVersion": "messages-v1",
        "messages": [{"role":"user","content":[
            {"image":{"format":_mime_to_nova(mime),
                      "source":{"bytes":base64.b64encode(image_bytes).decode()}}},
            {"text": prompt}
        ]}],
        "inferenceConfig": {"max_new_tokens": 1500, "temperature": 0}
    })
    try:
        resp     = _BEDROCK.invoke_model(modelId=model_id, body=body,
                                         accept="application/json",
                                         contentType="application/json")
        payload  = json.loads(resp["body"].read().decode("utf-8"))
        text_out = "".join(p.get("text","") for p in
                           payload.get("output",{}).get("message",{}).get("content",[])
                           if isinstance(p, dict))
        cleaned  = text_out.strip().strip("`")
        fb = cleaned.find("{"); lb = cleaned.rfind("}")
        if fb != -1 and lb != -1: cleaned = cleaned[fb:lb+1]
        # sanitize invalid \uXXXX escapes (Nova OCR of rupee symbol / Devanagari)
        cleaned = re.sub(r'\\u(?![0-9A-Fa-f]{4})', ' ', cleaned)
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            print("[WARN] JSON parse failed — using raw lines.")
            return {"lines":[l.strip() for l in text_out.splitlines() if l.strip()], "fields":{}}
    except ClientError as e:
        raise RuntimeError(f"AWS ClientError: {e}")
    except BotoCoreError as e:
        raise RuntimeError(f"Bedrock error: {e}")

# NEW: SIGNATURE PRESENCE CHECK (vision)
def detect_signature_presence(image_bytes, mime, model_id):
    """
    Uses the same Bedrock (Nova) vision model to check if a *handwritten signature*
    is present in the signature area. Returns True/False, defaulting to False if uncertain.
    """
    import base64
    prompt = (
        "You are a document vision checker. Look ONLY for a handwritten signature scribble "
        "in the designated signature area on a bank cheque (usually bottom-right near "
        "phrases like 'Authorised Signatory' or 'Please sign'). "
        "Ignore printed labels or text; confirm the presence of an actual handwritten stroke/signature.\n\n"
        "Respond strictly as JSON with a single boolean field:\n"
        "{\"signature_present\": true|false}\n"
        "No explanations, no markdown."
    )
    body = json.dumps({
        "schemaVersion": "messages-v1",
        "messages": [{"role":"user","content":[
            {"image":{"format":_mime_to_nova(mime),
                      "source":{"bytes":base64.b64encode(image_bytes).decode()}}},
            {"text": prompt}
        ]}],
        "inferenceConfig": {"max_new_tokens": 50, "temperature": 0}
    })
    try:
        resp     = _BEDROCK.invoke_model(modelId=model_id, body=body,
                                         accept="application/json",
                                         contentType="application/json")
        payload  = json.loads(resp["body"].read().decode("utf-8"))
        text_out = "".join(p.get("text","") for p in
                           payload.get("output",{}).get("message",{}).get("content",[])
                           if isinstance(p, dict)).strip()
        fb = text_out.find("{"); lb = text_out.rfind("}")
        if fb != -1 and lb != -1:
            try:
                obj = json.loads(text_out[fb:lb+1])
                return bool(obj.get("signature_present", False))
            except Exception:
                pass
        low = text_out.lower()
        if "true" in low or "yes" in low: return True
        if "false" in low or "no" in low: return False
        return False
    except Exception:
        return False

# ------- BANK↔IFSC consistency mapping & checker (NEW) -------
_BANK_IFSC_PREFIXES = [
    (re.compile(r'\bHDFC\b', re.I),    'HDFC'),
    (re.compile(r'\bICICI\b', re.I),   'ICIC'),
    (re.compile(r'\bSTATE\s*BANK\s*OF\s*INDIA\b|\bSBI\b', re.I), 'SBIN'),
    (re.compile(r'\bBANK\s*OF\s*BARODA\b|\bBOB\b', re.I),        'BARB'),
    (re.compile(r'\bAXIS\b', re.I),    'UTIB'),
    (re.compile(r'\bKOTAK\b', re.I),   'KKBK'),
    (re.compile(r'\bYES\b', re.I),     'YESB'),
    (re.compile(r'\bPUNJAB\s*NATIONAL\b|\bPNB\b', re.I), 'PUNB'),
    (re.compile(r'\bCANARA\b', re.I),  'CNRB'),
    (re.compile(r'\bUNION\s*BANK\b', re.I), 'UBIN'),
    (re.compile(r'\bIDBI\b', re.I),    'IBKL'),
    (re.compile(r'\bFEDERAL\b', re.I), 'FDRL'),
]

def _expected_ifsc_prefix_for_bank(bank_name):
    if not bank_name: return None
    for pat, code in _BANK_IFSC_PREFIXES:
        if pat.search(bank_name): return code
    return None
# -------------------------------------------------------------

# PHOENIX / COD HELPERS
def _to_date(v):
    if not v: return None
    if isinstance(v, datetime): return v.date()
    if isinstance(v, date) and not isinstance(v, datetime): return v
    r = try_parse_date(str(v))
    if r:
        try: return datetime.strptime(r, "%Y-%m-%d").date()
        except Exception: pass
    return None

def _to_decimal_str(v):
    if v is None: return None
    s = re.sub(r"[^\d.\-]", "", str(v).replace(",",""))
    if not s: return None
    try: return str(Decimal(s).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError): return None

def _to_ts(v):
    if v is None: return None
    if isinstance(v, datetime): return v
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try: return datetime.strptime(str(v).strip(), fmt)
        except Exception: pass
    return datetime.now()

def phoenix_connect():
    return phoenixdb.connect(url=PHOENIX_URL, autocommit=True,
                             auth=HTTPBasicAuth(PHOENIX_USER, PHOENIX_PASS),
                             serialization="PROTOBUF", verify=PHOENIX_VERIFY)

def ensure_table_exists(conn):
    ddl = (
        'CREATE TABLE "Cheque_data" ('
        '"Cheque number"       VARCHAR NOT NULL,'
        '"Account number"      VARCHAR NOT NULL,'
        '"Date on Cheque"      DATE,'
        '"Payee name"          VARCHAR,'
        '"Amount in Numbers"   VARCHAR,'
        '"Amount in words"     VARCHAR,'
        '"bank name"           VARCHAR,'
        '"IFSC code"           VARCHAR,'
        '"image name"          VARCHAR NOT NULL,'
        '"image uploaded time" TIMESTAMP,'
        'CONSTRAINT pk PRIMARY KEY ("Cheque number","Account number","image name"))'
    )
    cur = conn.cursor()
    try:
        if RECREATE_TABLE:
            try: cur.execute('DROP TABLE IF EXISTS "Cheque_data"')
            except Exception: pass
        try:
            cur.execute(ddl)
            print("[COD] Cheque_data table ready.")
        except Exception as e:
            if "already exists" not in str(e).lower(): raise
            print("[COD] Cheque_data already exists — skipping CREATE.")
    finally:
        try: cur.close()
        except Exception: pass

def upsert_into_cod(conn, fields):
    img  = str(fields.get("image_name") or "UNKNOWN")
    chq  = str(fields.get("cheque_number") or f"UNREADABLE_{img}").strip()
    acct = str(fields.get("account_number") or f"UNREADABLE_{img}").strip()
    values = (
        chq, acct,
        _to_date(fields.get("cheque_date")),
        fields.get("payee_name") or None,
        _to_decimal_str(fields.get("amount_numbers")),
        fields.get("amount_words") or None,
        fields.get("bank_name") or None,
        fields.get("ifsc_code") or None,
        img,
        _to_ts(fields.get("image_uploaded_time"))
    )
    sql = ('UPSERT INTO "Cheque_data"('
           '"Cheque number","Account number","Date on Cheque","Payee name",'
           '"Amount in Numbers","Amount in words","bank name",'
           '"IFSC code","image name","image uploaded time")'
           ' VALUES (?,?,?,?,?,?,?,?,?,?)')
    cur = conn.cursor()
    try:
        cur.execute(sql, values)
        print(f"[COD] UPSERT OK  cheque={chq}  account={acct}  image={img}")
    finally:
        try: cur.close()
        except Exception: pass

# MERGE MODEL + FALLBACK FIELDS
def merge_fields(model, fallback):
    out = {}
    for k in ['cheque_number','cheque_date','payee_name','amount_numbers',
              'amount_words','bank_name','account_number','ifsc_code']:
        mv = (model or {}).get(k)
        if isinstance(mv, str) and not mv.strip(): mv = None
        out[k] = mv if mv is not None else fallback.get(k)
    if out.get("cheque_date"):
        out["cheque_date"] = try_parse_date(str(out["cheque_date"])) or fallback.get("cheque_date")
    date_d = re.sub(r'\D','', str(out.get("cheque_date") or fallback.get("cheque_date") or ""))
    raw_a  = re.sub(r'\D','', str(out.get("amount_numbers") or "").replace(",",""))
    if raw_a and (raw_a == date_d or (len(raw_a)==6 and re.match(r'^[1-8]\d{5}$', raw_a))):
        out["amount_numbers"] = fallback.get("amount_numbers")
    if _try_ddmmyyyy(re.sub(r'\D','', str(out.get("cheque_number") or ""))):
        out["cheque_number"] = fallback.get("cheque_number")
    ifsc = normalize_ifsc(out.get("ifsc_code"))
    out["ifsc_code"] = ifsc if ifsc else fallback.get("ifsc_code")
    out["image_name"]             = fallback.get("image_name")
    out["image_uploaded_time"]    = fallback.get("image_uploaded_time")
    out["amount_number_raw_line"] = fallback.get("amount_number_raw_line")
    return out


# AMOUNT WORDS → NUMBER CONVERTER
_ONES  = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,
           'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'thirteen':13,
           'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,
           'nineteen':19,'twenty':20,'thirty':30,'forty':40,'fifty':50,
           'sixty':60,'seventy':70,'eighty':80,'ninety':90}
_SCALE = {'hundred':100,'thousand':1000,'lakh':100000,'lac':100000,
          'lakhs':100000,'lacs':100000,'crore':10000000,'crores':10000000}

def words_to_number(text):
    """Convert Indian English amount words to integer. Returns None if unparseable."""
    if not text: return None
    t = re.sub(r'(?i)\bonly\b', '', text)
    t = re.sub(r'[ऀ-ॿ]+', '', t)
    t = re.sub(r'[^a-zA-Z0-9\s]', ' ', t).lower().strip()
    if not t: return None
    tokens = t.split()
    current = 0; result = 0
    for tok in tokens:
        if tok in _ONES:
            current += _ONES[tok]
        elif tok == 'hundred':
            current = current * 100 if current else 100
        elif tok in _SCALE and tok != 'hundred':
            if current == 0: current = 1
            result += current * _SCALE[tok]
            current = 0
    result += current
    return result if result > 0 else None

def validate_amount_format(fields, raw_lines=None):
    """Three fraud checks: /- suffix, Only in words, words==numbers."""
    reasons   = []
    amt_num   = fields.get("amount_numbers")
    amt_words = (fields.get("amount_words") or "").strip()
    raw_lines = raw_lines or []
    raw_line  = fields.get("amount_number_raw_line") or ""

    if amt_num is not None:
        has_slash = bool(re.search(r'[\d,]+\s*/[-–]?', raw_line))
        if not has_slash:
            for l in raw_lines:
                if re.search(r'[₹\$]', l) and re.search(r'[\d,]+\s*/[-–]?', l):
                    has_slash = True; break
        if not has_slash:
            reasons.append("Amount in Numbers missing '/-' suffix")

    if amt_num is not None or amt_words:
        has_only = False
        for l in raw_lines:
            if re.search(r'(?i)\bonly\b', re.sub('[\u0900-\u097F]+', '', l)):
                has_only = True; break
        if not has_only and amt_words:
            has_only = bool(re.search(r'(?i)\bonly\b',
                            re.sub('[\u0900-\u097F]+', '', amt_words)))
        if not has_only:
            reasons.append("Amount in words missing 'Only'")

    if not reasons and amt_num is not None and amt_words:
        nd = _to_decimal_str(amt_num)
        nw = words_to_number(amt_words)
        if nd and nw and abs(float(nd) - float(nw)) > 0.99:
            reasons.append(
                f"Amount mismatch: numbers={nd} "
                f"but words='{amt_words}' (={nw})"
            )
    return reasons

def audit_fields(fields, raw_lines=None):
    """Return (missing_fields list, fraud_reasons list)."""
    missing = []
    def blank(v): return v is None or (isinstance(v, str) and not v.strip())
    if blank(fields.get("cheque_number")):                    missing.append("Cheque number")
    if _to_date(fields.get("cheque_date")) is None:           missing.append("Date on Cheque")
    if blank(fields.get("payee_name")):                       missing.append("Payee name")
    if _to_decimal_str(fields.get("amount_numbers")) is None: missing.append("Amount in Numbers")
    if blank(fields.get("amount_words")):                     missing.append("Amount in words")
    if blank(fields.get("bank_name")):                        missing.append("bank name")
    acct = re.sub(r'\D','', fields.get("account_number") or "")
    if not (6 <= len(acct) <= 20):                            missing.append("Account number")
    else: fields["account_number"] = acct
    ifsc = normalize_ifsc(fields.get("ifsc_code"))
    if not ifsc:                                              missing.append("IFSC code")
    else: fields["ifsc_code"] = ifsc
    fraud_reasons = validate_amount_format(fields, raw_lines or [])
    return missing, fraud_reasons

# MAIN BATCH
try:
    image_files = list_image_files(s3_path)
    if not image_files:
        print(f"No images found under {s3_path}"); spark.stop(); sys.exit(0)

    print(f"Found {len(image_files)} image(s) under {s3_path}")
    # cod_conn = phoenix_connect()
    # ensure_table_exists(cod_conn)   # COMMENTED OUT (no COD DDL for now)

    processed = upserts_ok = 0
    flagged = []; frauds = []; failures = []; seen_pks = set()

    for image_file in image_files:
        processed += 1
        print(f"\n{'='*70}\n[{processed}/{len(image_files)}] {image_file}\n{'='*70}")
        try:
            mime, _ = mimetypes.guess_type(image_file)
            mime    = mime or "image/jpeg"

            img_bytes   = read_bytes(image_file)
            bedrock_out = invoke_bedrock(img_bytes, mime, bedrock_model_id)

            seen_l = set(); text_lines = []
            for t in (bedrock_out.get("lines",[]) if isinstance(bedrock_out,dict) else []):
                if t not in seen_l: text_lines.append(t); seen_l.add(t)

            model_fields = (bedrock_out.get("fields") or {}) if isinstance(bedrock_out,dict) else {}
            fallback     = parse_cheque_fields(text_lines, image_name=os.path.basename(image_file))
            fields       = merge_fields(model_fields, fallback)

            print("\n---- Raw Lines ----")
            for t in text_lines: print(t)
            log = dict(fields); log["amount_numbers"] = _to_decimal_str(fields.get("amount_numbers"))
            print("\n---- Merged Output ----")
            print(json.dumps(log, indent=2, default=str))

            missing, fraud_reasons = audit_fields(fields, text_lines)

            # Signature presence
            signature_present = detect_signature_presence(img_bytes, mime, bedrock_model_id)
            if not signature_present:
                fraud_reasons.append("Signature missing on cheque")

            # Stale cheque (> 90 days)
            chq_dt = _to_date(fields.get("cheque_date"))
            if chq_dt:
                days_old = (datetime.now().date() - chq_dt).days
                if days_old > 90:
                    fraud_reasons.append(f"Stale cheque: date {chq_dt} older than 3 months")

            # Bank name vs IFSC bank code mismatch
            bank_name = fields.get("bank_name")
            ifsc_code = fields.get("ifsc_code")
            if bank_name and ifsc_code:
                exp_prefix = _expected_ifsc_prefix_for_bank(bank_name)
                if exp_prefix:
                    if not ifsc_code.upper().startswith(exp_prefix):
                        fraud_reasons.append(
                            f"Bank/IFSC mismatch: bank '{bank_name}' vs IFSC '{ifsc_code}' (expected prefix '{exp_prefix}')"
                        )

            if missing:
                print(f"[AUDIT] Missing fields : {', '.join(missing)} — upserting with NULL")
                flagged.append(f"{os.path.basename(image_file)} :: {', '.join(missing)}")

            if fraud_reasons:
                for r in fraud_reasons:
                    print(f"[FRAUD] {r}")
                frauds.append(f"{os.path.basename(image_file)} :: {' | '.join(fraud_reasons)}")

            pk = (str(fields.get("cheque_number","")).strip(),
                  str(fields.get("account_number","")).strip(),
                  str(fields.get("image_name","")).strip())
            if pk in seen_pks:
                print("[WARN] Duplicate PK — row will be overwritten")
            seen_pks.add(pk)

            # upsert_into_cod(cod_conn, fields)   # COMMENTED OUT (no COD writes)
            # upserts_ok += 1

        except Exception as e:
            msg = f"{os.path.basename(image_file)}: {repr(e)}"
            print(f"[ERROR] {msg}", file=sys.stderr)
            failures.append(msg)

    # try: cod_conn.close()
    # except Exception: pass

    print(f"\n{'#'*70}\nBATCH SUMMARY\n{'#'*70}")
    print(f"Total images     : {len(image_files)}")
    print(f"Processed        : {processed}")
    # print(f"UPSERT success   : {upserts_ok}")
    print(f"Distinct PK rows : {len(seen_pks)}")
    print(f"Flagged (NULLs)  : {len(flagged)}")
    print(f"Fraud detected   : {len(frauds)}")
    print(f"Failures         : {len(failures)}")
    if flagged:
        print("\nFlagged — upserted with NULL in:")
        for f in flagged: print(f"  - {f}")
    if frauds:
        print("\nFraud details (amount mismatch / missing suffix / words-numbers mismatch):")
        for f in frauds: print(f"  - {f}")
    if failures:
        print("\nFailure details:")
        for f in failures: print(f"  - {f}")
    print("#"*70)

except Exception as e:
    print(f"Fatal Error: {e}", file=sys.stderr); raise
finally:
    spark.stop()
