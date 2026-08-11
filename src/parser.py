"""
共用字段解析模块 (CNN分支 与 YOLO分支 共用)。

输入格式约定: parse_receipt(lines)
    lines: List[Tuple[str, box]]
        text: 识别出的文字
        box:  numpy array shape(4,2)，四点坐标[左上,右上,右下,左下]；也可以传 None(不参与位置计算，
              退化成按list原始顺序当作从上到下的顺序)
    两条pipeline(DBNet+CRNN / YOLO)只要把各自识别出的(文字,坐标)整理成这个格式传进来即可。

输出: {"company": str, "address": str, "date": str, "total": str}  (抽取不到则为空字符串)
"""
import re

import numpy as np

MONEY_RE = re.compile(r"\d{1,3}(?:,\d{3})*\.\d{2}")
PURE_MONEY_RE = re.compile(r"^(RM)?\s*\d{1,3}(?:,\d{3})*\.\d{2}$", re.IGNORECASE)

DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})\b"),   # 24/06/2018, 24-06-18
    re.compile(r"\b(\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2})\b"),     # 2018-06-24
]

# 马来西亚收据地址常见关键词/州名
ADDRESS_KEYWORDS = [
    "JALAN", "JLN", "LORONG", "PERSIARAN", "TAMAN", "BANDAR", "KAMPUNG", "KG ",
    "LEVEL", "TINGKAT", "LOT ", "NO.", "NO ", "BLOCK", "BLOK", "UNIT",
    "SELANGOR", "JOHOR", "PENANG", "PULAU PINANG", "PERAK", "KEDAH", "MELAKA",
    "NEGERI SEMBILAN", "PAHANG", "KELANTAN", "TERENGGANU", "SABAH", "SARAWAK",
    "KUALA LUMPUR", "PUTRAJAYA", "PERLIS",
]
POSTCODE_RE = re.compile(r"\b\d{5}\b")

# 明显不是地址/公司名的行，扫描地址时跳过但不打断扫描
NOISE_LINE_RE = re.compile(
    r"(GST\s*ID|TEL\s*NO|PHONE|FAX|LICENSEE|WEBSITE|EMAIL)", re.IGNORECASE
)
# 公司注册号，如 "(65351-M)"，纯数字容易被误判成邮编，单独排除
REG_NUMBER_RE = re.compile(r"^\(?\d{4,}[-–][A-Z]\)?$")
# 遇到这些代表已经进入商品明细区，地址扫描应该停止
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
    """统一成 [(text, box_or_None, y_center), ...] 并按 y_center 从上到下排序。"""
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
            y_center = i  # 没有坐标就用原始顺序当"高度"
        normalized.append((text, box, y_center))
    normalized.sort(key=lambda x: x[2])
    return normalized


def extract_company(norm_lines):
    """公司名：从上往下第一条"看起来像名字"的行(含字母、长度>=3、不是纯噪声行)。"""
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
    地址：在公司名之后的前 max_scan 行里，挑出符合地址关键词/邮编特征的行，按顺序拼接。
    遇到"进入商品明细"的标志行就停止扫描。
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
        is_address_line = POSTCODE_RE.search(text) or any(kw in upper for kw in ADDRESS_KEYWORDS)
        if is_address_line:
            parts.append(text)

    return " ".join(parts)


def extract_date(norm_lines):
    for text, _, _ in norm_lines:
        for pattern in DATE_PATTERNS:
            m = pattern.search(text)
            if m:
                return m.group(1)
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

        # 情况1: 金额就在本行内 (如 "TOTAL RM12.80")
        m = MONEY_RE.search(text)
        if m:
            return m.group(0)

        # 情况2: 用box位置找"同一行右侧"的金额
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

        # 情况3: 退化成"紧接着的下一行"(纯金额格式)
        if idx + 1 < len(norm_lines):
            next_text = norm_lines[idx + 1][0].strip()
            if PURE_MONEY_RE.match(next_text):
                m3 = MONEY_RE.search(next_text)
                if m3:
                    return m3.group(0)

    return ""


def parse_receipt(lines):
    """
    lines: List[Tuple[str, box]] 或 List[str]
    返回: {"company": str, "address": str, "date": str, "total": str}
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