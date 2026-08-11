import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from crnn_utils import encode_text


def load_labels(labels_file):
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


def augment_line_image(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]

    if random.random() < 0.3:
        angle = random.uniform(-3, 3)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)

    if random.random() < 0.25:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    if random.random() < 0.3:
        alpha = random.uniform(0.7, 1.3)
        beta = random.uniform(-30, 30)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if random.random() < 0.2:
        noise = np.random.normal(0, 8, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


class CRNNDataset(Dataset):
    def __init__(self, crops_dir, labels_file, char_to_idx, img_height=32, max_width=280,
                 augment=False):
        self.crops_dir = crops_dir
        self.items = load_labels(labels_file)
        self.char_to_idx = char_to_idx
        self.img_height = img_height
        self.max_width = max_width
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        crop_name, text = self.items[idx]
        path = os.path.join(self.crops_dir, crop_name)
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

        if self.augment:
            img = augment_line_image(img)

        img = resize_pad_width(img, self.img_height, self.max_width)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  
        img_tensor = torch.from_numpy(img).unsqueeze(0)  

        target = encode_text(text, self.char_to_idx)
        return img_tensor, torch.LongTensor(target), text


def crnn_collate_fn(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = torch.cat([b[1] for b in batch])
    target_lengths = torch.LongTensor([len(b[1]) for b in batch])
    texts = [b[2] for b in batch]
    return images, targets, target_lengths, texts