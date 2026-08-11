"""
CRNN 共用工具函数：字符表加载/编解码、CTC贪心解码、字符错误率(CER)计算。
训练(train_crnn.py)和推理(crnn_recognizer.py)都用这里的编解码逻辑，保证一致。
"""
import os
import editdistance


def load_alphabet(alphabet_file):
    """
    读取 alphabet.txt，返回 (char_to_idx, idx_to_char)。
    index 0 保留给 CTC 的 blank 符号，实际字符从 index 1 开始。
    """
    chars = []
    with open(alphabet_file, "r", encoding="utf-8") as f:
        for line in f:
            ch = line[:-1] if line.endswith("\n") else line
            if ch == "":
                continue
            chars.append(ch)

    char_to_idx = {ch: i + 1 for i, ch in enumerate(chars)}  # 0 留给 blank
    idx_to_char = {i + 1: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char


def encode_text(text, char_to_idx):
    """字符串 -> index列表(用于CTCLoss的target)。未知字符直接跳过(理论上不应出现,alphabet是从训练集统计的)。"""
    return [char_to_idx[ch] for ch in text if ch in char_to_idx]


def ctc_greedy_decode(indices, idx_to_char):
    """
    CTC贪心解码：连续重复的index合并成一个，再去掉blank(index 0)。
    indices: 一维list/array，模型每个时间步argmax后的类别index序列。
    """
    chars = []
    prev = -1
    for idx in indices:
        idx = int(idx)
        if idx != prev and idx != 0:
            chars.append(idx_to_char.get(idx, ""))
        prev = idx
    return "".join(chars)


def compute_cer(pred: str, gt: str) -> float:
    """字符错误率 = 编辑距离 / GT长度。GT为空时特殊处理避免除0。"""
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return editdistance.eval(pred, gt) / len(gt)