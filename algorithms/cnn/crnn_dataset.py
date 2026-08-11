"""
CRNN 训练用 Dataset。
读取 prepare_crnn_data.py 生成的裁剪文字行图片 + labels.txt，
统一resize/pad到固定宽度(config.CRNN_IMG_MAX_WIDTH)，转灰度单通道。
文字标签用 alphabet 编码成 index 序列，给 CTCLoss 用。
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from crnn_utils import encode_text


def load_labels(labels_file):
    """读取 labels.txt (每行: 图片名\\t文字)，返回 [(crop_name, text), ...]"""
    items = []
    if not os.path.exists(labels_file):
        return items
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            items.append((parts[0], parts[1]))
    return items


def resize_pad_width(gray_img: np.ndarray, target_height: int, max_width: int) -> np.ndarray:
    """
    输入已经是target_height高度的灰度图(crop_quad裁剪时已统一高度)。
    宽度 <= max_width: 右侧补白(255)到max_width。
    宽度 >  max_width: 直接resize压缩到max_width(极少数超长行,轻微形变可接受)。
    """
    h, w = gray_img.shape[:2]
    if h != target_height:
        gray_img = cv2.resize(gray_img, (int(w * target_height / h), target_height))
        w = gray_img.shape[1]

    if w <= max_width:
        canvas = np.full((target_height, max_width), 255, dtype=np.uint8)
        canvas[:, :w] = gray_img
        return canvas
    else:
        return cv2.resize(gray_img, (max_width, target_height))


class CRNNDataset(Dataset):
    def __init__(self, crops_dir, labels_file, char_to_idx, img_height=32, max_width=280):
        self.crops_dir = crops_dir
        self.items = load_labels(labels_file)
        self.char_to_idx = char_to_idx
        self.img_height = img_height
        self.max_width = max_width

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        crop_name, text = self.items[idx]
        path = os.path.join(self.crops_dir, crop_name)
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

        img = resize_pad_width(img, self.img_height, self.max_width)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # 归一化到[-1,1]
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)

        target = encode_text(text, self.char_to_idx)
        return img_tensor, torch.LongTensor(target), text


def crnn_collate_fn(batch):
    """
    batch: [(img_tensor, target_tensor, text_str), ...]
    图片宽度已经在Dataset里统一pad到max_width了，可以直接stack。
    target长度不一致，拼成一维长向量 + 每条的长度列表 (CTCLoss要求的格式)。
    """
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = torch.cat([b[1] for b in batch])
    target_lengths = torch.LongTensor([len(b[1]) for b in batch])
    texts = [b[2] for b in batch]
    return images, targets, target_lengths, texts