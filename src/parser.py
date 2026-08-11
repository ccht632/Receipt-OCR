"""
Common Field Parsing Module (Shared by CNN Branch and YOLO Branch)

Input Format Convention: parse_receipt(lines)
    lines: List[Tuple[str, box]]
        text: Recognized text
        box: numpy array with shape (4, 2), four-point coordinates in order:
             [Top-Left, Top-Right, Bottom-Right, Bottom-Left];
             Can also pass None (does not participate in position calculation,
             falls back to treating list order as top-to-bottom sequence)

    Both pipelines (DBNet+CRNN / YOLO) only need to format their recognized
    (text, coordinates) into this structure before passing it in.

Output:
    {"company": str, "address": str, "date": str, "total": str}
    (Returns empty string for any field that cannot be extracted)
"""
import re

import editdistance
import numpy as np

MONEY_RE = re.compile(r"\d{1,3}(?:,\d{3})*\.\d{2}")
PURE_MONEY_RE = re.compile(r"^(RM)?\s*\d{1,3}(?:,\d{3})*\.\d{2}$", re.IGNORECASE)

MONTH_NAMES = (r"JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|JUN(?:E)?|"
               r"JUL(?:Y)?|AUG(?:UST)?|SEP(?:T|TEMBER)?|OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?")

DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})\b"),   # 24/06/2018, 24-06-18
    re.compile(r"\b(\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2})\b"),     # 2018-06-24
    re.compile(rf"\b(\d{{1,2}}\s+(?:{MONTH_NAMES})\s+\d{{2,4}})\b", re.IGNORECASE),  # 19 JULY 2026
    re.compile(rf"\b((?:{MONTH_NAMES})\s+\d{{1,2}},?\s+\d{{2,4}})\b", re.IGNORECASE),  # JULY 19, 2026
]

MONTH_ABBR = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
MONTH_FULL = ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE",
              "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]
# Compare the edit distance of the fully recognized word (without truncation) with both the "abbreviation" and the "full month name".
# Take the closest one globally -- this also helps with disambiguation due to length information (e.g., "JALY" is closer to JULY than JAN).
_MONTH_CANDIDATES = [(abbr, abbr) for abbr in MONTH_ABBR] + [(full, abbr) for full, abbr in zip(MONTH_FULL, MONTH_ABBR)]

# Backup: Any structure of "numbers, three to nine alphanumeric characters", with the alphanumeric part using edit distance for fuzzy matching of the month.
# (Tolerates one character recognition error, such as AUG often being recognized as AUO/AWG on thermal receipts)
FUZZY_DATE_RE = re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{2,4})\b")


def _fuzzy_match_month(word: str):
    """
    Calculate edit distance between the FULL recognized word and
    month abbreviations / full month names. Tolerate up to 1 character error.
    Return None if no match is found.
    """
    word = word.upper()
    best_abbr, best_dist = None, 2
    for candidate, abbr in _MONTH_CANDIDATES:
        d = editdistance.eval(word, candidate)
        if d < best_dist:
            best_abbr, best_dist = abbr, d
    return best_abbr if best_dist <= 1 else None

# Common keywords/state names for receipt addresses in Malaysia
ADDRESS_KEYWORDS = [
    "JALAN", "JLN", "LORONG", "PERSIARAN", "TAMAN", "BANDAR", "KAMPUNG", "KG ",
    "LEVEL", "TINGKAT", "LOT ", "NO.", "NO ", "BLOCK", "BLOK", "UNIT",
    "SELANGOR", "JOHOR", "PENANG", "PULAU PINANG", "PERAK", "KEDAH", "MELAKA",
    "NEGERI SEMBILAN", "PAHANG", "KELANTAN", "TERENGGANU", "SABAH", "SARAWAK",
    "KUALA LUMPUR", "PUTRAJAYA", "PERLIS",
]
POSTCODE_RE = re.compile(r"\b\d{5}\b")

# Lines that are clearly not addresses/company names should be skipped during address scanning without interrupting the scan.
NOISE_LINE_RE = re.compile(
    r"(GST\s*ID|TEL\s*NO|PHONE|FAX|LICENSEE|WEBSITE|EMAIL|REG\s*NO)", re.IGNORECASE
)
# Company registration number, such as "(65351-M)", is easily misinterpreted as a postal code due to its purely numerical format; therefore, it should be excluded.
REG_NUMBER_RE = re.compile(r"^\(?\d{4,}[-–][A-Z]\)?$")
# When you encounter these numbers, it means you've entered the product details area; address scanning should stop.
STOP_ADDRESS_SCAN_RE = re.compile(
    r"(TAX INVOICE|^INVOICE|QTY|ITEM|RECEIPT NO|ORD\s*#|TABLE)", re.IGNORECASE
)

TOTAL_KEYWORD_RE = re.compile(r"\bTOTAL\b", re.IGNORECASE)
TOTAL_EXCLUDE_RE = re.compile(r"(SUB\s*TOTAL|SUBTOTAL)", re.IGNORECASE)


def _box_bounds(box):
    if box is None:
        return None
    box = np.asarray(box)
    return box[:, 0].min(), box[:, 1].min(), box[:, 0].max(), box[:, 1].max()


def _normalize_lines(lines):
    """
    Unify format into [(text, box_or_None, y_center), ...]
    and sort entries from top to bottom by y_center.
    """
    normalized = []
    for i, item in enumerate(lines):
        if isinstance(item, (tuple, list)) and len(item) == 2:
            text, box = item
        else:
            text, box = item, None
        text = (text or "").strip()
        if not text:
            continue
        if box is not None:
            _, ymin, _, ymax = _box_bounds(box)
            y_center = (ymin + ymax) / 2
        else:
            y_center = i  # If no coordinates are available, use the original order as the "height".
        normalized.append((text, box, y_center))
    normalized.sort(key=lambda x: x[2])
    return normalized


def extract_company(norm_lines):
    """
    Company Name: The first line that "appears to be a name" when scanning top-to-bottom
    (contains letters, length ≥ 3, and is not a pure noise line).
    """
    skip_words = ("TAX INVOICE", "RECEIPT", "INVOICE", "SIMPLIFIED")
    for text, _, _ in norm_lines:
        letters = sum(c.isalpha() for c in text)
        if letters < 3:
            continue
        if text.strip().upper() in skip_words:
            continue
        if NOISE_LINE_RE.search(text):
            continue
        return text
    return ""


def extract_address(norm_lines, company_text, max_scan=10):
    """
    Address: Scan within the first `max_scan` lines following the company name.
    Select lines matching address keywords or postal code patterns,
    and concatenate them in order.
    Stop scanning once a line marking the start of item details is encountered.
    """
    parts = []
    started = False
    scanned = 0
    for text, _, _ in norm_lines:
        if not started:
            if text == company_text:
                started = True
            continue

        if scanned >= max_scan:
            break
        scanned += 1

        if STOP_ADDRESS_SCAN_RE.search(text):
            break
        if NOISE_LINE_RE.search(text) or REG_NUMBER_RE.match(text.strip()):
            continue

        upper = text.upper()
        is_address_line = (
            POSTCODE_RE.search(text)
            or any(kw in upper for kw in ADDRESS_KEYWORDS)
            or text.count(",") >= 2  # Malaysian addresses commonly use a comma-separated format: "street, district, city".
                                       # A fallback situation where OCR misidentifies place names (e.g., KUALA->KUALE), causing keyword matching failures.
        )
        if is_address_line:
            parts.append(text)

    return " ".join(parts)


def extract_date(norm_lines):
    for text, _, _ in norm_lines:
        for pattern in DATE_PATTERNS:
            m = pattern.search(text)
            if m:
                return m.group(1)

    # No match found in strict mode -> Use fuzzy month matching as a fallback (tolerates OCR misinterpreting the month abbreviation by one character).
    for text, _, _ in norm_lines:
        m = FUZZY_DATE_RE.search(text)
        if m:
            day, month_word, year = m.groups()
            matched_month = _fuzzy_match_month(month_word)
            if matched_month:
                return f"{day} {matched_month} {year}"
    return ""


def extract_total(norm_lines):
    """
    找含独立单词TOTAL的行(排除SUBTOTAL)。
    金额优先在同一行找；同一行没有就找"同一水平行(box)在右边的相邻行"；
    没有box信息就退化成"紧接着的下一行"(适配label在上、金额在下堆叠的排版)。
    """
    for idx, (text, box, _) in enumerate(norm_lines):
        if not TOTAL_KEYWORD_RE.search(text):
            continue
        if TOTAL_EXCLUDE_RE.search(text):
            continue

        # Scenario 1: The amount is within this bank's balance (e.g., "TOTAL RM12.80")
        m = MONEY_RE.search(text)
        if m:
            return m.group(0)

        # Scenario 2: Use the box position to find the amount "on the right side of the same row".
        if box is not None:
            label_bounds = _box_bounds(box)
            best = None
            best_dx = None
            for j, (t2, b2, _) in enumerate(norm_lines):
                if j == idx or b2 is None:
                    continue
                x2min, y2min, x2max, y2max = _box_bounds(b2)
                lx_min, ly_min, lx_max, ly_max = label_bounds
                overlap = min(ly_max, y2max) - max(ly_min, y2min)
                row_height = max(ly_max - ly_min, y2max - y2min)
                same_row = row_height > 0 and overlap / row_height > 0.4
                if same_row and x2min >= lx_max and PURE_MONEY_RE.match(t2.strip()):
                    dx = x2min - lx_max
                    if best is None or dx < best_dx:
                        best, best_dx = t2, dx
            if best:
                m2 = MONEY_RE.search(best)
                if m2:
                    return m2.group(0)

        # Case 3: Degenerates into "the next line immediately following" (pure amount format)
        if idx + 1 < len(norm_lines):
            next_text = norm_lines[idx + 1][0].strip()
            if PURE_MONEY_RE.match(next_text):
                m3 = MONEY_RE.search(next_text)
                if m3:
                    return m3.group(0)

    return ""


def parse_receipt(lines):
    """
    lines: List[Tuple[str, box]] or List[str]
    Return: {"company": str, "address": str, "date": str, "total": str}
    """
    norm_lines = _normalize_lines(lines)

    company = extract_company(norm_lines)
    address = extract_address(norm_lines, company)
    date = extract_date(norm_lines)
    total = extract_total(norm_lines)

    return {
        "company": company,
        "address": address,
        "date": date,
        "total": total,
    }