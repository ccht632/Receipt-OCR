import os
import editdistance


def load_alphabet(alphabet_file):
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
    return [char_to_idx[ch] for ch in text if ch in char_to_idx]


def ctc_greedy_decode(indices, idx_to_char):
    chars = []
    prev = -1
    for idx in indices:
        idx = int(idx)
        if idx != prev and idx != 0:
            chars.append(idx_to_char.get(idx, ""))
        prev = idx
    return "".join(chars)


def compute_cer(pred: str, gt: str) -> float:
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return editdistance.eval(pred, gt) / len(gt)