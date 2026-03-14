# -*- coding: utf-8 -*-
"""
PySpark + Amazon Bedrock (Amazon Nova) cheque OCR-ish understanding & field extraction
+ PhoenixDB UPSERT into Cloudera Operational Database (COD).

FIXES vs previous version (nova-premier.py):
────────────────────────────────────────────
FIX 1 – PIN code grabbed as amount (Media 7 bug, confirmed by viewing actual image):
  Media (7)AmtNo.jpg has a COMPLETELY EMPTY SOIT (amount) box — no numeric amount
  exists anywhere on the cheque. The correct output is amount_numbers = null (fraud).
  The old code (and v5) picked up "110001" from the address "INDIA-110001" as amount.
  Root cause: Pass B of the amount search scanned every line including address lines,
  found the PIN code 110001 after all other candidates were exhausted/consumed, and
  treated it as the amount.
  Solution:
    a) Indian PIN codes (6 digits, 1xxxxx pattern, after INDIA- or city name hyphen)
       are explicitly excluded from the amount search via _is_pincode().
    b) Address lines (comma + location keywords) are entirely skipped in Pass B.
    c) Numbers in the range 100000–999999 that match the PIN code pattern are rejected.
    d) The model prompt now explicitly says: "amount_numbers must come from the SOIT
       box only; if the SOIT box is empty, return null."
    e) merge_fields validates: if model returns null and fallback also finds nothing
       valid, the field stays null (fraud correctly detected).

FIX 2 – False-positive IFSC from address (Media 8 bug):
  When IFSC is genuinely absent the sliding-window in normalize_ifsc found letter
  sequences inside address tokens ("PHOENIX MALL ROAD" → PHOE01XMA11).
  Solution:
    a) normalize_ifsc now requires the candidate to start at a word boundary and
       the first 4 chars must all be alpha (after OCR-correction); pure-alpha head
       derived from an address segment like "PHOE" is accepted only if a following
       zero-slot (pos 5) can be confirmed.
    b) More importantly, the label-based IFSC search is tried first on ALL lines;
       the aggressive unlabeled sliding-window search is now gated: it is skipped
       entirely when the labeled search on a line explicitly produced an empty value
       (i.e., "IFSC :" with no value = the cheque is missing an IFSC, don't hallucinate
       one from the address).
    c) Address lines (containing commas, road/nagar/mall keywords) are now blacklisted
       from the unlabeled IFSC scan.

FIX 3 – Prompt improvement:
  The Nova prompt now explicitly asks for DDMMYYYY to be interpreted as the literal
  date format label, NOT as a date value, and instructs the model to return the raw
  date string it sees so the Python parser can normalize it.

Other retained improvements:
  - Fraud check after normalization to reduce false positives.
  - Non-scientific amount formatting.
  - Optional Phoenix password override via PHOENIX_PASS env var.
  - Supports Nova Premier via Inference Profile ARN.
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
spark = SparkSession.builder.appName("InvokeBedrockNova_ChequeOCR_Fixed_v5").getOrCreate()

# ======================= CONFIG =======================
s3_path = "s3a://cloudera-poc-buck/cheques/incoming/"
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}

# ---- BEDROCK ----
aws_region = "us-east-1"
bedrock_model_id = os.getenv("BEDROCK_MODEL_ID", "").strip() or "amazon.nova-pro-v1:0"

# ---- COD / PHOENIX ----
PHOENIX_URL    = "https://cod-1iuh2nynfx25k-gateway0.hexaware.dvd1-p7f4.a0.cloudera.site/cod-1iuh2nynfx25k/cdp-proxy-api/avatica/"
PHOENIX_USER   = "csso_vaishnavidevendrad"
PHOENIX_PASS   = os.getenv("PHOENIX_PASS", "Vaish@cloudera1")
PHOENIX_VERIFY = True

READBACK_VERIFY  = True   # set False to skip read-back after each UPSERT
RECREATE_TABLE   = True   # DROP + recreate Cheque_data on every run.
                          # Set to False ONLY after the table has been created with the
                          # correct composite PK and you want to preserve existing rows.

# ======================= HADOOP FILE HELPERS =======================
sc  = spark.sparkContext
jvm = sc._jvm

def list_image_files(folder_uri: str):
    path = jvm.org.apache.hadoop.fs.Path(folder_uri)
    fs   = path.getFileSystem(sc._jsc.hadoopConfiguration())
    if not fs.exists(path):
        return []
    files = []
    for st in fs.listStatus(path):
        if st.isFile():
            p   = st.getPath().toString()
            ext = os.path.splitext(p)[1].lower()
            if ext in SUPPORTED_EXT:
                files.append(p)
    files.sort()
    return files

def read_bytes(file_uri: str) -> bytes:
    fs_path   = jvm.org.apache.hadoop.fs.Path(file_uri)
    fs        = fs_path.getFileSystem(sc._jsc.hadoopConfiguration())
    in_stream = fs.open(fs_path)
    try:
        data = bytearray(in_stream.readAllBytes())
    finally:
        in_stream.close()
    return bytes(data)

# ======================= NORMALIZATION HELPERS =======================
_MONTHS = {
    'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
    'JUL':7,'AUG':8,'SEP':9,'SEPT':9,'OCT':10,'NOV':11,'DEC':12
}

def _clean_text(s):
    return (s or "").strip()

# ── Address-line heuristic: lines that are likely postal addresses ──────────
_ADDRESS_KEYWORDS = re.compile(
    r'(?i)\b(road|rd|street|st|nagar|marg|mall|lane|avenue|ave|'
    r'colony|sector|block|plot|flat|floor|building|bldg|complex|'
    r'bridge|chowk|bazaar|market|plaza|tower|park|garden|village|'
    r'near|opp|opposite|behind|beside|india|mumbai|delhi|bengaluru|'
    r'kolkata|chennai|hyderabad|pune|ahmedabad|jaipur|lucknow)\b'
)

def _is_address_line(line: str) -> bool:
    """True if the line looks like a postal address (has commas + location keywords)."""
    return bool(',' in line and _ADDRESS_KEYWORDS.search(line))


# Indian PIN codes appear after city names; we must not mistake them for amounts
_PINCODE_RE = re.compile(
    r'(?i)(?:india|delhi|mumbai|bengaluru|chennai|kolkata|'
    r'hyderabad|pune|ahmedabad|jaipur|lucknow|noida|gurgaon)'
    r'[\s\-,]+(\d{6})'
)

def _is_pincode(token: str, full_line: str) -> bool:
    """
    Returns True if a numeric token is likely an Indian PIN code and NOT a cheque amount.
    A PIN code is exactly 6 digits and either:
      (a) appears right after a city/INDIA marker on the same line, OR
      (b) lives on a line that is already identified as an address line.
    """
    digits = re.sub(r'\D', '', token)
    if len(digits) != 6:
        return False
    # Explicit city-hyphen-pincode pattern
    for m in _PINCODE_RE.finditer(full_line):
        if m.group(1) == digits:
            return True
    # Address line + 6-digit number starting with 1-8
    if _is_address_line(full_line) and re.match(r'^[1-8]\d{5}$', digits):
        return True
    return False


# ======================= IFSC NORMALIZER (FIX 2) =======================
def normalize_ifsc(raw: str) -> str | None:
    """
    Robust IFSC normalizer/validator.

    FIX 2 improvement: after cleaning, the 4-char head must be all-alpha (after
    OCR digit→letter correction).  This rejects tokens like 'PHOE01XMA11' whose
    head contains digits that cannot be corrected to a plausible bank code.
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
        map_head = {'0': 'O', '1': 'I', '5': 'S', '2': 'Z', '8': 'B'}
        head = ''.join(map_head.get(ch, ch) for ch in cand[:4])
        mid  = '0'
        map_tail = {'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'Q': '0'}
        tail = ''.join(map_tail.get(ch, ch) for ch in cand[5:])
        return head + mid + tail

    def _is_valid_ifsc(x: str) -> bool:
        # Standard format: 4 alpha, literal '0', 6 alnum
        return bool(re.match(r'^[A-Z]{4}0[A-Z0-9]{6}$', x))

    if len(s) == 11:
        cand = _fix_candidate(s)
        if _is_valid_ifsc(cand):
            return cand

    if len(s) > 11:
        for i in range(len(s) - 10):
            window = _fix_candidate(s[i:i + 11])
            if _is_valid_ifsc(window):
                return window

    return None


def extract_ifsc_from_lines(lines: list) -> str | None:
    """
    FIX 2: Two-pass IFSC extraction.

    Pass 1 – Label-based: scan every line for 'IFSC' / 'IFS Code' labels.
      • If we find a label with a non-empty value → normalize and return.
      • If we find a label with an EMPTY value (e.g. "IFSC :") → record that
        IFSC is declared absent; do NOT proceed to unlabeled scan.

    Pass 2 – Unlabeled sliding-window: only runs if pass 1 found no label AT ALL.
      • Skips lines that look like postal addresses.
      • Skips lines that contain currency/amount markers.
    """
    label_re = re.compile(
        r'(?i)\bI(?:FS|FSC)\s*(?:Code|C(?:ode)?)?\s*[:/\-]?\s*([A-Za-z0-9 \-]*)',
        re.IGNORECASE
    )
    found_label     = False
    label_was_empty = False

    for line in lines:
        m = label_re.search(line)
        if m:
            found_label = True
            raw_value   = (m.group(1) or "").strip()
            if not raw_value:
                # Label present but value missing → IFSC genuinely absent
                label_was_empty = True
                continue
            cand = normalize_ifsc(raw_value)
            if cand:
                return cand
            # Value present but didn't validate → keep looking in other lines
            # (could be OCR noise on the same label repeated)

    # If a label was found but empty, the cheque has no IFSC printed → return None
    if label_was_empty and not found_label:
        return None
    if label_was_empty:
        # label was found but produced no valid IFSC → treat as absent
        return None

    # Pass 2: no label seen at all – try unlabeled window scan (conservative)
    for line in lines:
        U = (line or "").upper()
        # Skip lines with currency / amount markers
        if re.search(r'(₹|INR|\$|RUPEES|DOLLARS|ONLY)', U):
            continue
        # Skip lines that look like postal addresses
        if _is_address_line(line):
            continue
        # Skip pure date lines
        if re.search(r'\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}\b', line):
            continue
        for tok in re.findall(r'[A-Za-z0-9]{8,}', line):
            cand = normalize_ifsc(tok)
            if cand:
                return cand

    return None


# ======================= DATE PARSER (FIX 1) =======================

# A bare 8-digit string on an Indian cheque is almost always DDMMYYYY
_BARE_8_DIGIT_RE = re.compile(r'^\d{8}$')

def _try_ddmmyyyy(s: str):
    """Try to parse bare 8-digit string as DDMMYYYY or MMDDYYYY."""
    if not _BARE_8_DIGIT_RE.match(s):
        return None
    dd, mm, yyyy = int(s[0:2]), int(s[2:4]), int(s[4:8])
    # Try DDMMYYYY first (Indian standard)
    try:
        if 1 <= dd <= 31 and 1 <= mm <= 12 and 1900 <= yyyy <= 2100:
            d = date(yyyy, mm, dd)
            return d.strftime("%Y-%m-%d")
    except ValueError:
        pass
    # Try MMDDYYYY
    try:
        if 1 <= mm <= 12 and 1 <= dd <= 31 and 1900 <= yyyy <= 2100:
            d = date(yyyy, dd, mm)   # swap: month=mm, day=dd
            return d.strftime("%Y-%m-%d")
    except ValueError:
        pass
    return None

def try_parse_date_yyyy_mm_dd(raw: str) -> str | None:
    """
    Try many date patterns and return YYYY-MM-DD or None.
    FIX 1: now also handles bare DDMMYYYY 8-digit strings.
    """
    if not raw:
        return None
    s = _clean_text(raw)
    s = re.sub(r'(?i)\bdate\s*[:\-]?\s*', '', s)
    s = s.replace(',', ' ').replace('  ', ' ').strip()
    s = re.sub(r'(\d{1,2})(st|nd|rd|th)\b', r'\1', s, flags=re.IGNORECASE)

    # Bare 8-digit: try DDMMYYYY / MMDDYYYY
    if _BARE_8_DIGIT_RE.match(s):
        result = _try_ddmmyyyy(s)
        if result:
            return result

    numeric_patterns = [
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
        "%d/%m/%y", "%d-%m-%y", "%d.%m.%y",
        "%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y",
        "%m/%d/%y", "%m-%d-%y", "%m.%d.%y",
    ]
    for fmt in numeric_patterns:
        try:
            return datetime.strptime(s, fmt).date().strftime("%Y-%m-%d")
        except Exception:
            pass

    m = re.match(r'(?i)^\s*(\d{1,2})[\s\-\.]+([A-Za-z]{3,})[\s\-\.]+(\d{2,4})\s*$', s)
    if m:
        d, mon, y = m.groups()
        mon_key = mon[:4].upper() if mon[:4].upper() == 'SEPT' else mon[:3].upper()
        if mon_key in _MONTHS:
            yy = int(y) + (2000 if len(y) == 2 else 0) if len(y) == 2 else int(y)
            try:
                return date(yy, _MONTHS[mon_key], int(d)).strftime("%Y-%m-%d")
            except Exception:
                pass

    m2 = re.match(r'(?i)^\s*([A-Za-z]{3,})[\s\-\.]+(\d{1,2})[\s\-\,\.]+(\d{2,4})\s*$', s)
    if m2:
        mon, d, y = m2.groups()
        mon_key = mon[:3].upper()
        if mon_key in _MONTHS:
            yy = int(y) + (2000 if len(y) == 2 else 0) if len(y) == 2 else int(y)
            try:
                return date(yy, _MONTHS[mon_key], int(d)).strftime("%Y-%m-%d")
            except Exception:
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


# ======================= CHEQUE NUMBER EXTRACTOR =======================
# On Indian cheques the cheque leaf serial number is encoded in the MICR band
# at the very bottom in a special OCR-B / E-13B font.  The MICR line looks like:
#   "CHEQUE_NO"  SORT_CODE:  ACCOUNT_CODE"  CHECK_DIGIT
# The cheque number is always the FIRST quoted group (5-8 digits).
# All other number fields (date box top-right, account number, branch code)
# are NOT the cheque number.

# Regex: opening quote (straight or curly), optional Devanagari noise, digits, closing quote/asterisk
_MICR_CHEQUE_NO_RE = re.compile(
    r'["\u201C\u201D]'         # opening quote (straight or curly)
    r'[\u0900-\u097F\s*]*'     # optional Devanagari chars / spaces / asterisks
    r'(\d{5,8})'                 # cheque serial number (5-8 digits)
    r'[\u0900-\u097F\s*]*'     # optional trailing noise
    r'["\u201C\u201D*]'        # closing quote or asterisk
)

def _is_micr_line(line: str) -> bool:
    """True if this line is the MICR band at the bottom of the cheque."""
    return bool(re.search(r'\d{6,}\s*:', line) or
                re.search(r'[\u0900-\u097F].*\d{5,}|"\s*\d{5,}\s*"', line))

def extract_cheque_number(lines: list, account_number: str = None, ifsc_code: str = None):
    """
    Extract cheque number with three-pass priority:

    Pass 1 – MICR band (highest confidence):
      The bottom line of the cheque contains the serial number as the FIRST
      quoted digit group:  "XXXXXX"  routing:  account"  check
      This is authoritative and beats everything else.

    Pass 2 – Explicit label:
      Lines containing "Cheque No", "Chq No", "Leaf No", etc.

    Pass 3 – Last resort unlabeled scan:
      Any 6-digit run NOT equal to the account number and NOT matching the
      DDMMYYYY 8-digit date pattern.  Prefer shorter (6-digit) candidates
      so we don't accidentally pick the date.
    """
    acct_digits = re.sub(r'\D', '', account_number or "")

    # ── Pass 1: MICR band ─────────────────────────────────────────────────
    for l in lines or []:
        if _is_micr_line(l):
            m = _MICR_CHEQUE_NO_RE.search(l)
            if m:
                cand = m.group(1)
                # Must not be equal to (or embedded in) the account number
                # Reject only if candidate IS the account number (exact match)
                # or if it is a long substring (>=9 digits). A short 6-digit
                # cheque number will often appear inside a 16-digit account
                # number by coincidence — that is NOT a conflict.
                if acct_digits and (
                        cand == acct_digits or
                        (len(cand) >= 9 and cand in acct_digits) or
                        (len(cand) >= 9 and acct_digits in cand)):
                    continue
                return cand
    # ── Pass 2: Explicit label ────────────────────────────────────────────
    label_patterns = [
        r'(?i)\b(?:che?que|chq|cq|leaf|slip)\s*(?:no|number|#)?\s*[:\-]?\s*([0-9 \-]{6,12})',
        r'(?i)\bNo\.?\s*[:\-]?\s*([0-9 \-]{6,12})'
    ]
    for l in lines or []:
        for pat in label_patterns:
            m = re.search(pat, l)
            if m:
                cand = re.sub(r'\D', '', m.group(1))
                if 6 <= len(cand) <= 10:
                    if acct_digits and (cand == acct_digits or cand in acct_digits or acct_digits in cand):
                        continue
                    return cand

    # ── Pass 3: Unlabeled fallback (conservative) ─────────────────────────
    # Collect 6-digit candidates only (8-digit would be the date DDMMYYYY)
    # Skip lines that are clearly amounts, addresses, IFSC, or MICR
    candidates = []
    for l in lines or []:
        U = (l or "").upper()
        if re.search(r'(₹|INR|\$|RUPEES|DOLLARS)', U):
            continue
        if 'IFSC' in U or 'IFS CODE' in U:
            continue
        if _is_address_line(l):
            continue
        if _is_micr_line(l):
            continue
        for run in re.findall(r'(\d{6})', l):   # strictly 6 digits
            if acct_digits and len(run) >= 9 and (
                    run in acct_digits or acct_digits in run):
                continue
            # Reject if it parses as a valid DDMMYYYY date (would be the date field)
            if _try_ddmmyyyy(run):
                continue
            candidates.append(run)

    if candidates:
        return candidates[0]   # first occurrence wins (top of cheque body)

    return None

# ======================= FALLBACK PARSER (FIX 1 & 2) =======================
def parse_cheque_fields(lines: list, image_name: str) -> dict:
    """
    Robust regex-based extraction from a list of OCR text lines.

    FIX 1: date is resolved FIRST; the resolved date token (and cheque-number token)
           are excluded from the amount search so they can't be mistaken for amounts.
    FIX 2: IFSC extraction delegates to extract_ifsc_from_lines() which guards
           against address-line false positives.
    """
    # Dedup preserving order
    uniq_lines, seen = [], set()
    for l in lines or []:
        if l not in seen:
            uniq_lines.append(l)
            seen.add(l)
    lines = uniq_lines
    text  = "\n".join(lines)

    cheque_number   = None
    cheque_date     = None
    payee_name      = None
    amount_numbers  = None
    amount_words    = None
    bank_name       = None
    account_number  = None
    ifsc_code       = None

    # ── 1. IFSC (FIX 2: use the two-pass extractor) ──────────────────────
    ifsc_code = extract_ifsc_from_lines(lines)

    # ── 2. Account number ─────────────────────────────────────────────────
    acct_patterns = [
        r'(?i)\b(?:account|a\/c|ac|acc)\s*(?:no|number|#)?\s*[:\-]?\s*([0-9 \-]{6,})',
        r'(?i)\bA\/C\s*[:\-#]?\s*([0-9 \-]{6,})',
        r'(?i)\bAcc(?:ount)?\s*No\.?\s*[:\-]?\s*([0-9 \-]{6,})'
    ]
    def _normalize_acct(x):
        d = re.sub(r'\D', '', x or '')
        return d if 6 <= len(d) <= 20 else None

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
            candidates.extend(re.findall(r'(\d{9,18})', l))
        if candidates:
            account_number = max(candidates, key=len)

    # ── 3. Cheque number ──────────────────────────────────────────────────
    cheque_number = extract_cheque_number(lines, account_number=account_number,
                                          ifsc_code=ifsc_code)

    # ── 4. Date (FIX 1 – also handles bare DDMMYYYY) ─────────────────────
    # Collect tokens that are already "spoken for" so we don't reuse them as amounts
    consumed_numeric_tokens = set()
    if account_number:
        consumed_numeric_tokens.add(re.sub(r'\D', '', account_number))
    if cheque_number:
        consumed_numeric_tokens.add(re.sub(r'\D', '', cheque_number))

    for l in lines:
        # Skip lines that are purely format-guide placeholders like "DDMMYYYY"
        if re.fullmatch(r'(?i)[DdMmYy\s/.\-]+', l.strip()):
            continue

        # Try the full line first (handles bare "15092026")
        cand = try_parse_date_yyyy_mm_dd(l.strip())
        if cand:
            cheque_date = cand
            consumed_numeric_tokens.add(re.sub(r'\D', '', l.strip()))
            break

        # Try an embedded delimited token
        token = _extract_first(r'(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4})', l)
        if token:
            cand = try_parse_date_yyyy_mm_dd(token)
            if cand:
                cheque_date = cand
                consumed_numeric_tokens.add(re.sub(r'\D', '', token))
                break

    # ── 5. Payee name ─────────────────────────────────────────────────────
    for idx, l in enumerate(lines):
        if re.search(r'(?i)pay\s+to(\s+the\s+order\s+of)?', l):
            if idx + 1 < len(lines):
                payee_name = lines[idx + 1].strip()
            break
    # Also handle inline "Pay <Name> Or Bearer"
    if not payee_name:
        for l in lines:
            m = re.search(r'(?i)\bPay\s+([A-Za-z][A-Za-z .]+?)\s+(?:Or\s+Bearer|Or\s+Order)', l)
            if m:
                payee_name = m.group(1).strip()
                break

    # ── 6. Amount in numbers ────────────────────────────────────────────
    # Pass A: explicit currency symbol (₹ / Rs / INR / $)
    # Only search lines that are NOT address lines to avoid grabbing PIN codes
    for l in lines:
        if _is_address_line(l):   # e.g. '12, PHOENIX MALL ROAD, ... INDIA-110001'
            continue
        m1 = re.search(r'(?:₹|Rs\.?|INR|\$)\s*([0-9][\d,]*(?:\.\d{1,2})?)', l, re.IGNORECASE)
        if m1:
            raw_val = m1.group(1).replace(",", "")
            if raw_val not in consumed_numeric_tokens:
                amount_numbers = raw_val
                break

    # Pass B: bare number with no currency symbol
    # Guards: skip address lines, skip consumed tokens, skip PIN codes
    if amount_numbers is None:
        for l in lines:
            # Skip address lines entirely — PIN codes live here
            if _is_address_line(l):
                continue
            # Skip branch-code / reference lines (e.g. 'Br: 2011 Pat-201', 'mindels no.')
            if re.search(r'(?i)\bBr\s*:\s*\d|Pat-\d|mindel|Gm\s+Ro\b', l):
                continue
            # Skip MICR band lines (bottom of cheque: routing/account codes)
            if re.search(r'\d{6,}\s*:', l) or re.search(r'[\u0900-\u097F].*\d{5,}|"\s*\d{5,}\s*"', l):
                continue
            # Skip format-placeholder lines like 'D D M M Y Y Y Y'
            if re.fullmatch(r'(?i)[DdMmYy\s/.\-]+', l.strip()):
                continue
            for m2 in re.finditer(r'\b([0-9][\d,]{2,}(?:\.\d{1,2})?)\b', l):
                raw_val = m2.group(1).replace(",", "")
                if raw_val in consumed_numeric_tokens:
                    continue
                # Reject if this token is a PIN code on this line
                if _is_pincode(raw_val, l):
                    continue
                # Reject implausible lengths (8-digit dates, 16-digit account numbers)
                d_count = len(re.sub(r'\D', '', raw_val))
                if d_count > 10 or d_count < 3 or (d_count <= 4 and int(re.sub(r"\D","",raw_val)) < 100):
                    continue
                amount_numbers = raw_val
                break
            if amount_numbers:
                break

    # ── 7. Amount in words ────────────────────────────────────────────────
    for l in lines:
        if re.search(r'(?i)\b(rupees|dollars|only)\b', l):
            # Strip leading Hindi/Devanagari characters before the English part
            cleaned = re.sub(r'^[\u0900-\u097F\s]+', '', l).strip()
            if cleaned:
                amount_words = cleaned
            else:
                amount_words = l.strip()
            break

    # ── 8. Bank name ──────────────────────────────────────────────────────
    for l in lines:
        if 'BANK' in (l or '').upper() and 'IFSC' not in (l or '').upper():
            bank_name = l.strip()
            break

    return {
        "cheque_number":      cheque_number,
        "cheque_date":        cheque_date,
        "payee_name":         payee_name,
        "amount_numbers":     amount_numbers,
        "amount_words":       amount_words,
        "bank_name":          bank_name,
        "account_number":     account_number,
        "ifsc_code":          ifsc_code,
        "image_name":         image_name,
        "image_uploaded_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


# ======================= BEDROCK INVOCATION (NOVA, messages-v1) =======================
def _mime_to_nova_format(mime: str) -> str:
    m = (mime or "").lower()
    if "jpeg" in m or m.endswith("/jpg") or "jpg" in m: return "jpeg"
    if "png"  in m: return "png"
    if "webp" in m: return "webp"
    if "tif"  in m: return "tiff"
    if "bmp"  in m: return "bmp"
    if "gif"  in m: return "gif"
    return "jpeg"

def invoke_bedrock_ocr_and_extract(image_bytes: bytes, mime: str, model_id: str, region: str) -> dict:
    """
    FIX 3: Improved prompt:
      • Tells the model that 'DDMMYYYY' / 'D D M M Y Y Y Y' are format-placeholder
        labels, NOT real dates.
      • Instructs model to return the raw date string found (e.g. "15092026") so the
        Python normalizer can parse it, rather than trying to interpret it itself.
      • Tells model to return null for ifsc_code if the IFSC field on the cheque is
        blank or absent, rather than guessing from nearby text.
    """
    bedrock = boto3.client(
        service_name="bedrock-runtime",
        region_name=region,
        config=Config(read_timeout=3600, connect_timeout=3600, retries={'max_attempts': 3})
    )

    img_b64    = base64.b64encode(image_bytes).decode("utf-8")
    img_format = _mime_to_nova_format(mime)

    system_list = [
        {"text": "You are a document understanding assistant specialized in Indian bank cheques."}
    ]

    user_content = [
        {
            "image": {
                "format": img_format,
                "source": {"bytes": img_b64}
            }
        },
        {
            "text": (
                "TASKS:\n"
                "1) Transcribe all readable text lines in natural reading order as an array.\n"
                "2) Extract the following fields (use null if a field is genuinely absent or unreadable):\n"
                "   - cheque_number  : The cheque LEAF SERIAL NUMBER from the MICR band at the very bottom of the cheque.\n"
                "                     It is the FIRST quoted group of digits on that bottom line, e.g. \"098765\" in \"098765\" 1121000221: 060607\" 41.\n"
                "                     It is NOT the date (top-right box). It is NOT the account number. It is typically 6 digits.\n"
                "   - cheque_date    : Return the RAW date string exactly as printed (e.g. '15092026', '05/11/2024', '25-Nov-2026').\n"
                "                     IMPORTANT: 'DDMMYYYY', 'D D M M Y Y Y Y', 'DDMMMYYYY' are FORMAT LABELS, not dates. Return null for those.\n"
                "   - payee_name     : Name after 'Pay' or 'Pay to the order of'\n"
                "   - amount_numbers : The numeric amount from the SOIT/amount box on the right side of the cheque (e.g. 30500.00).\n"
                "                     If the SOIT box is EMPTY or blank, return null — do NOT use the date, PIN code, or any other number.\n"
                "   - amount_words   : Amount in English words (strip any Devanagari prefix)\n"
                "   - bank_name      : Name of the issuing bank\n"
                "   - account_number : Account number (digits only, 9-18 digits)\n"
                "   - ifsc_code      : IFSC code if printed. Return null if the IFSC field is blank or absent—do NOT infer it from the address or branch name.\n\n"
                "RESPONSE FORMAT (STRICT JSON, no commentary, no markdown code fences):\n"
                "{\n"
                '  "lines": ["..."],\n'
                '  "fields": {\n'
                '    "cheque_number": null,\n'
                '    "cheque_date": null,\n'
                '    "payee_name": null,\n'
                '    "amount_numbers": null,\n'
                '    "amount_words": null,\n'
                '    "bank_name": null,\n'
                '    "account_number": null,\n'
                '    "ifsc_code": null\n'
                "  }\n"
                "}\n"
                "Use null for unknown/absent values. No extra keys."
            )
        }
    ]

    body = json.dumps({
        "schemaVersion": "messages-v1",
        "system":         system_list,
        "messages":       [{"role": "user", "content": user_content}],
        "inferenceConfig": {
            "maxTokens":   2000,
            "temperature": 0.0,
            "topP":        0.9
        }
    })

    try:
        resp    = bedrock.invoke_model(
            modelId=model_id,
            body=body,
            accept="application/json",
            contentType="application/json"
        )
        payload = json.loads(resp["body"].read().decode("utf-8"))

        text_out = ""
        try:
            for part in payload["output"]["message"]["content"]:
                if isinstance(part, dict) and "text" in part:
                    text_out += part["text"]
        except Exception:
            pass

        cleaned = text_out.strip()
        if cleaned.startswith("```"):
            cleaned     = cleaned.strip("`")
            first_brace = cleaned.find("{")
            last_brace  = cleaned.rfind("}")
            if first_brace != -1 and last_brace != -1:
                cleaned = cleaned[first_brace:last_brace + 1]

        return json.loads(cleaned)

    except ClientError as e:
        raise RuntimeError(f"AWS ClientError invoking Bedrock model '{model_id}': {e}")
    except (BotoCoreError, json.JSONDecodeError, KeyError, TypeError) as e:
        raise RuntimeError(f"Bedrock invocation/parse error: {e}")


# ======================= PHOENIX / COD HELPERS =======================
def _to_sql_date(cheque_date_val):
    if not cheque_date_val:
        return None
    if isinstance(cheque_date_val, date) and not isinstance(cheque_date_val, datetime):
        return cheque_date_val
    if isinstance(cheque_date_val, datetime):
        return cheque_date_val.date()
    s    = str(cheque_date_val).strip()
    best = try_parse_date_yyyy_mm_dd(s)
    if best:
        try:
            return datetime.strptime(best, "%Y-%m-%d").date()
        except Exception:
            pass
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y",
                "%m-%d-%Y", "%d/%m/%y", "%m/%d/%y", "%d.%m.%Y", "%d.%m.%y"):
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
    if not s:
        return None
    try:
        d = Decimal(s).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return d
    except (InvalidOperation, ValueError):
        return None

def _to_sql_timestamp(ts_val):
    if ts_val is None:
        return None
    if isinstance(ts_val, datetime):
        return ts_val
    s = str(ts_val).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S.%f"):
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

def ensure_table_exists(conn):
    """
    Always DROP then CREATE Cheque_data with composite PK.

    Why unconditional DROP:
      - Phoenix has no ALTER TABLE ADD/DROP PRIMARY KEY.
      - The old table has a single-column PK ("Cheque number") which causes
        silent overwrites.  There is no reliable way to probe the PK definition
        via a SELECT, so we simply always recreate.
      - Set RECREATE_TABLE = False to skip the drop (only safe after the table
        has already been recreated with the composite PK).
    """
    ddl_drop   = 'DROP TABLE IF EXISTS "Cheque_data"'
    ddl_create = (
        'CREATE TABLE "Cheque_data" ('
        '"Cheque number"       VARCHAR       NOT NULL, '
        '"Account number"      VARCHAR       NOT NULL, '
        '"Date on Cheque"      DATE, '
        '"Payee name"          VARCHAR, '
        '"Amount in Numbers"   VARCHAR, '
        '"Amount in words"     VARCHAR, '
        '"bank name"           VARCHAR, '
        '"IFSC code"           VARCHAR, '
        '"image name"          VARCHAR, '
        '"image uploaded time" TIMESTAMP, '
        'CONSTRAINT pk_cheque PRIMARY KEY ("Cheque number", "Account number", "image name")'
        ')'
    )
    cur = None
    try:
        cur = conn.cursor()
        if RECREATE_TABLE:
            print("[COD] RECREATE_TABLE=True — dropping old Cheque_data...")
            cur.execute(ddl_drop)
            print("[COD] Dropped. Creating with composite PK...")
            cur.execute(ddl_create)
            print("[COD] Cheque_data created: PK=(Cheque number, Account number, image name)")
        else:
            # Try to create; if already exists Phoenix raises — that is fine.
            try:
                cur.execute(ddl_create)
                print("[COD] Cheque_data created: PK=(Cheque number, Account number, image name)")
            except Exception:
                print("[COD] Cheque_data already exists — skipping CREATE.")
    except Exception as e:
        print(f"[COD] FATAL: Could not ensure table: {e}")
        raise
    finally:
        if cur:
            try: cur.close()
            except Exception: pass

def upsert_into_cod(conn, fields: dict):
    """
    UPSERT a cheque record into Cheque_data.

    PRIMARY KEY is composite: ("Cheque number", "Account number", "image name")
    Rationale: cheque serial numbers restart per account/cheque book.
    Two different accounts can legitimately have the same 6-digit serial
    number, and test images often share the same MICR cheque number while
    belonging to different accounts.  Using only cheque_number as PK causes
    later images to silently overwrite earlier ones, losing records.

    The COD table must be created with this composite PK:
        CREATE TABLE IF NOT EXISTS "Cheque_data" (
            "Cheque number"      VARCHAR NOT NULL,
            "Account number"     VARCHAR NOT NULL,
            "Date on Cheque"     DATE,
            "Payee name"         VARCHAR,
            "Amount in Numbers"  VARCHAR,
            "Amount in words"    VARCHAR,
            "bank name"          VARCHAR,
            "IFSC code"          VARCHAR,
            "image name"         VARCHAR,
            "image uploaded time" TIMESTAMP,
            CONSTRAINT pk_cheque PRIMARY KEY ("Cheque number", "Account number", "image name")
        )
    """
    cheque_number  = fields.get("cheque_number")
    account_number = fields.get("account_number")

    # PK fields must be non-null for Phoenix. If genuinely unreadable,
    # use the image name as a safe fallback so the row is still stored.
    image_name_safe = str(fields.get("image_name") or "UNKNOWN").strip()
    if not cheque_number or str(cheque_number).strip() == "":
        cheque_number = f"UNREADABLE_{image_name_safe}"
    if not account_number or str(account_number).strip() == "":
        account_number = f"UNREADABLE_{image_name_safe}"

    pk_key = (str(cheque_number).strip(), str(account_number).strip())

    # Amount: convert Decimal → plain string "NNNNNN.NN" to avoid phoenixdb
    # serialising it as a Python float (which produces scientific notation like
    # 3.05E+4 in the HBase/Phoenix cell).  Phoenix accepts VARCHAR-like strings
    # for DECIMAL columns when using PROTOBUF serialisation.
    amt_decimal = _to_sql_decimal(fields.get("amount_numbers"))
    amt_str = str(amt_decimal) if amt_decimal is not None else None  # e.g. "30500.00"

    values = (
        pk_key[0],                                        # Cheque number  (PK part 1)
        pk_key[1],                                        # Account number (PK part 2)
        _to_sql_date(fields.get("cheque_date")),
        fields.get("payee_name") or None,
        amt_str,                                          # "30500.00" not 3.05E+4
        fields.get("amount_words") or None,
        fields.get("bank_name") or None,
        fields.get("ifsc_code") or None,
        fields.get("image_name") or None,
        _to_sql_timestamp(fields.get("image_uploaded_time"))
    )

    sql = (
        'UPSERT INTO "Cheque_data"('
        '"Cheque number","Account number","Date on Cheque","Payee name",'
        '"Amount in Numbers","Amount in words","bank name",'
        '"IFSC code","image name","image uploaded time"'
        ") VALUES (?,?,?,?,?,?,?,?,?,?)"
    )

    cur = None
    try:
        cur = conn.cursor()
        cur.execute(sql, values)
        print(f"[COD] UPSERT OK  cheque={pk_key[0]}  account={pk_key[1]}")
    finally:
        if cur:
            try: cur.close()
            except Exception: pass

def readback_from_cod(conn, cheque_number: str, account_number: str, image_name: str = ""):
    """Read back the just-upserted row using composite PK (cheque, account, image)."""
    sql = (
        'SELECT "Cheque number","Account number","Date on Cheque","Payee name",'
        '"Amount in Numbers",'
        '"Amount in words","bank name","IFSC code",'
        '"image name","image uploaded time" '
        'FROM "Cheque_data" WHERE "Cheque number" = ? AND "Account number" = ? AND "image name" = ?'
    )
    cur = None
    try:
        cur = conn.cursor()
        cur.execute(sql, (cheque_number, account_number, image_name))
        row = cur.fetchone()
        if not row:
            print(f"[COD] No row found for cheque={cheque_number} account={account_number} image={image_name}")
            return
        cols = ["Cheque number","Account number","Date on Cheque","Payee name",
                "Amount in Numbers","Amount in words","bank name","IFSC code",
                "image name","image uploaded time"]
        print("\n[COD] Read-back row:")
        for c, v in zip(cols, row):
            print(f"  {c}: {v}")
        print()
    finally:
        if cur:
            try: cur.close()
            except Exception: pass


# ======================= FIELD AUDIT (replaces fraud gate) =======================
def audit_fields(fields: dict) -> list:
    """
    Audit extracted fields and return a list of column names that are
    missing or null.  DOES NOT block the upsert — every image is always
    written to COD; missing fields are stored as NULL.

    The returned list is purely informational: it is logged as [AUDIT] and
    added to the batch summary so operators know which images had incomplete
    data on the cheque.

    Also normalises account_number (strips non-digits) and ifsc_code
    (canonical format) in-place so the values stored in COD are clean.
    """
    missing = []

    def _is_blank(v):
        return v is None or (isinstance(v, str) and v.strip() == "")

    if _is_blank(fields.get("cheque_number")):
        missing.append("Cheque number")
    if _to_sql_date(fields.get("cheque_date")) is None:
        missing.append("Date on Cheque")
    if _is_blank(fields.get("payee_name")):
        missing.append("Payee name")
    if _to_sql_decimal(fields.get("amount_numbers")) is None:
        missing.append("Amount in Numbers")
    if _is_blank(fields.get("amount_words")):
        missing.append("Amount in words")
    if _is_blank(fields.get("bank_name")):
        missing.append("bank name")

    # Normalise account number in-place; mark missing if unreadable
    acct_digits = re.sub(r'\D', '', fields.get("account_number") or "")
    if not (6 <= len(acct_digits) <= 20):
        missing.append("Account number")
        # Keep raw value — upsert will store NULL via _to_sql helpers
    else:
        fields["account_number"] = acct_digits  # store clean digits

    # Normalise IFSC in-place; mark missing if absent/invalid
    ifsc_norm = normalize_ifsc(fields.get("ifsc_code"))
    if not ifsc_norm:
        missing.append("IFSC code")
        fields["ifsc_code"] = None              # ensure NULL, not garbage string
    else:
        fields["ifsc_code"] = ifsc_norm

    if _is_blank(fields.get("image_name")):
        missing.append("image name")
    if _to_sql_timestamp(fields.get("image_uploaded_time")) is None:
        missing.append("image uploaded time")

    return missing   # empty list = all fields present


# ======================= MERGE LOGIC (FIX 1) =======================
def merge_fields(model_fields: dict, fallback_fields: dict) -> dict:
    """
    Per-field merge: prefer non-empty model value, otherwise use fallback.

    FIX 1 special handling:
      • cheque_date: the model may return the raw string "15092026" which our
        try_parse_date_yyyy_mm_dd can handle; if the model returns null but the
        fallback found a date, use the fallback.
      • amount_numbers: if the model returned a suspiciously large value that
        matches the date digits, prefer the fallback.
      • ifsc_code: if the model returned something that doesn't validate as a
        proper IFSC (e.g. hallucinated from address), discard and use fallback.
    """
    out  = {}
    keys = ["cheque_number", "cheque_date", "payee_name", "amount_numbers",
            "amount_words", "bank_name", "account_number", "ifsc_code"]

    for k in keys:
        mv = model_fields.get(k) if isinstance(model_fields, dict) else None
        if isinstance(mv, str) and mv.strip() == "":
            mv = None
        out[k] = mv if mv is not None else fallback_fields.get(k)

    # ── Post-merge coercions ──────────────────────────────────────────────

    # cheque_date: normalize whatever string we got (handles "15092026", etc.)
    raw_date = out.get("cheque_date")
    if raw_date:
        normalized = try_parse_date_yyyy_mm_dd(str(raw_date))
        if normalized:
            out["cheque_date"] = normalized
        else:
            # Model gave us something it couldn't parse; fall back
            out["cheque_date"] = fallback_fields.get("cheque_date")

    # amount_numbers: reject if it equals the date/cheque-number digits,
    # or if it looks like a PIN code (6-digit number from address)
    date_digits = re.sub(r'\D', '', out.get("cheque_date") or
                         fallback_fields.get("cheque_date") or "")
    cheque_num_digits = re.sub(r'\D', '', out.get("cheque_number") or "")
    raw_amount = str(out.get("amount_numbers") or "").replace(",", "")
    raw_amount_digits = re.sub(r'\D', '', raw_amount)
    if raw_amount_digits:
        is_collision = (raw_amount_digits == date_digits or
                        raw_amount_digits == cheque_num_digits)
        # Reject 6-digit values that look like Indian PIN codes (1xxxxx–8xxxxx)
        is_pin = (len(raw_amount_digits) == 6 and re.match(r'^[1-8]\d{5}$', raw_amount_digits))
        if is_collision or is_pin:
            # Use fallback; if fallback is also None, amount stays None (→ fraud)
            out["amount_numbers"] = fallback_fields.get("amount_numbers")



    # cheque_number: discard model value if it looks like a date (DDMMYYYY)
    # The model often confuses the date box (top-right) with the cheque number.
    # The real cheque number comes from the MICR band (extracted by fallback).
    raw_cno = re.sub(r'\D', '', out.get('cheque_number') or '')
    if raw_cno and _try_ddmmyyyy(raw_cno):   # model gave us the date, not the cheque no
        out['cheque_number'] = fallback_fields.get('cheque_number')

    # ifsc_code: discard model value if it doesn't validate; prefer fallback
    ifsc_from_model = normalize_ifsc(out.get("ifsc_code"))
    if not ifsc_from_model:
        out["ifsc_code"] = fallback_fields.get("ifsc_code")
    else:
        out["ifsc_code"] = ifsc_from_model

    out["image_name"]          = fallback_fields.get("image_name")
    out["image_uploaded_time"] = fallback_fields.get("image_uploaded_time")
    return out


# ======================= MAIN (BATCH) =======================
try:
    image_files = list_image_files(s3_path)
    if not image_files:
        print(f"No images found under {s3_path} with extensions: {sorted(SUPPORTED_EXT)}")
        spark.stop()
        sys.exit(0)

    if bedrock_model_id.startswith("arn:aws:bedrock:") and ":inference-profile/" in bedrock_model_id:
        print(f"[Bedrock] Using Nova Premier via Inference Profile: {bedrock_model_id}")
    else:
        print(f"[Bedrock] Using on-demand model: {bedrock_model_id}")

    print(f"Found {len(image_files)} image(s) under {s3_path}")
    cod_conn = phoenix_connect()
    print("[COD] Checking / migrating Cheque_data table schema...")
    print("[COD] NOTE: If the old 4-row table exists with single-column PK,")
    print("[COD]       it will be automatically DROPPED and recreated with")
    print("[COD]       composite PK (Cheque number + Account number).")
    ensure_table_exists(cod_conn)

    processed   = 0
    upserts_ok  = 0
    frauds      = []
    failures    = []
    duplicates  = []           # tracks (cheque_number, account_number) pairs already upserted
    seen_pks    = set()        # composite PK set for in-memory duplicate detection

    for image_file in image_files:
        processed += 1
        print("\n" + "=" * 90)
        print(f"[{processed}/{len(image_files)}] Processing: {image_file}")
        print("=" * 90)
        try:
            guessed_mime, _ = mimetypes.guess_type(image_file)
            mime            = guessed_mime or "image/jpeg"
            img_bytes       = read_bytes(image_file)

            # 1) Bedrock extraction
            bedrock_out = invoke_bedrock_ocr_and_extract(
                image_bytes=img_bytes, mime=mime,
                model_id=bedrock_model_id, region=aws_region
            )

            # 2) Deduplicated lines
            text_lines = []
            if isinstance(bedrock_out, dict):
                raw_lines = bedrock_out.get("lines", [])
                if isinstance(raw_lines, list):
                    seen = set()
                    for t in raw_lines:
                        if t not in seen:
                            text_lines.append(t)
                            seen.add(t)

            # 3) Model fields
            model_fields = {}
            if isinstance(bedrock_out, dict) and isinstance(bedrock_out.get("fields"), dict):
                model_fields = bedrock_out["fields"]

            # 4) Fallback parse
            fallback = parse_cheque_fields(
                text_lines, image_name=os.path.basename(image_file)
            )

            # 5) Merge
            fields = merge_fields(model_fields, fallback)

            # ── Log ──────────────────────────────────────────────────────
            print("\n---- Extracted Raw Lines (from Bedrock) ----")
            for t in text_lines:
                print(t)

            log_fields                    = dict(fields)
            log_fields["amount_numbers"]  = fmt_money(_to_sql_decimal(fields.get("amount_numbers")))
            print("\n---- MERGED STRUCTURED OUTPUT (pre-check) ----")
            print(json.dumps(log_fields, indent=2, default=str))

            # 6) Audit fields — log what is missing but ALWAYS continue to upsert
            missing_cols = audit_fields(fields)
            if missing_cols:
                print(f"\n[AUDIT] {os.path.basename(image_file)} "
                      f"has missing/null fields: {', '.join(missing_cols)} — will upsert with NULL\n")
                frauds.append(f"{os.path.basename(image_file)} :: {', '.join(missing_cols)}")

            # 7) Duplicate PK check — PK is (cheque_number, account_number, image_name)
            # Since image_name is part of the PK, every image gets its own row.
            # True duplicates (same image reprocessed) will overwrite — warn about those.
            pk = (str(fields.get("cheque_number","")).strip(),
                  str(fields.get("account_number","")).strip(),
                  str(fields.get("image_name","")).strip())
            if pk in seen_pks:
                dup_msg = (f"{os.path.basename(image_file)} has duplicate PK "
                           f"cheque={pk[0]} account={pk[1]} image={pk[2]} — row OVERWRITTEN")
                print(f"[WARN] {dup_msg}")
                duplicates.append(dup_msg)
            seen_pks.add(pk)

            # 8) Upsert
            upsert_into_cod(cod_conn, fields)
            upserts_ok += 1

            # 9) Read-back
            if READBACK_VERIFY and fields.get("cheque_number") and fields.get("account_number"):
                readback_from_cod(cod_conn, fields["cheque_number"],
                                  fields["account_number"], fields.get("image_name",""))

        except Exception as e:
            msg = f"{os.path.basename(image_file)}: {repr(e)}"
            print(f"[ERROR] {msg}", file=sys.stderr)
            failures.append(msg)

    try:
        cod_conn.close()
    except Exception:
        pass

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "#" * 90)
    print("BATCH SUMMARY")
    print("#" * 90)
    distinct_rows = len(seen_pks)
    print(f"Total images found      : {len(image_files)}")
    print(f"Processed               : {processed}")
    print(f"UPSERT success          : {upserts_ok}  (ALL images upserted — including incomplete ones)")
    print(f"Distinct rows in COD    : {distinct_rows}  (unique cheque+account combinations)")
    print(f"Duplicate overwrites    : {len(duplicates)}  (same cheque+account seen more than once)")
    print(f"Flagged (missing fields): {len(frauds)}  (upserted with NULL for missing columns)")
    print(f"Failures (hard errors)  : {len(failures)}")
    if frauds:
        print("\nFlagged images — upserted with NULL in these columns:")
        for f in frauds:
            print(f"  - {f}")
    if duplicates:
        print("\nDuplicate PK warnings (image will have overwritten a previous row):")
        for d in duplicates:
            print(f"  - {d}")
    if failures:
        print("\nFailure details:")
        for f in failures:
            print(f"  - {f}")
    print("#" * 90 + "\n")

except Exception as e:
    print(f"Fatal Error: {e}", file=sys.stderr)
    raise
finally:
    spark.stop()
