import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dbnet_gt import make_shrink_map, make_threshold_map


def letterbox_with_polygons(image, polygons, target_size):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = resized

    new_polygons = []
    for poly in polygons:
        new_polygons.append(poly * scale)
    return canvas, new_polygons


def parse_gt_file(gt_path):
    polygons, texts = [], []
    with open(gt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 8)
            if len(parts) < 9:
                continue
            try:
                coords = np.array([float(x) for x in parts[:8]], dtype=np.float32).reshape(4, 2)
            except ValueError:
                continue
            polygons.append(coords)
            texts.append(parts[8])
    return polygons, texts


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DBNetDataset(Dataset):
    def __init__(self, img_dir, gt_dir, name_list, input_size=640, shrink_ratio=0.4):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.names = [n for n in name_list if n]
        self.input_size = input_size
        self.shrink_ratio = shrink_ratio

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img_name = self.names[idx]
        stem = os.path.splitext(img_name)[0]

        data = np.fromfile(os.path.join(self.img_dir, img_name), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)

        polygons, _ = parse_gt_file(os.path.join(self.gt_dir, stem + ".txt"))

        image, polygons = letterbox_with_polygons(image, polygons, self.input_size)

        h = w = self.input_size
        prob_gt, prob_mask = make_shrink_map(polygons, h, w, self.shrink_ratio)
        thresh_gt, thresh_mask = make_threshold_map(polygons, h, w, self.shrink_ratio)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        img_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float()

        return {
            "image": img_tensor,
            "prob_gt": torch.from_numpy(prob_gt).unsqueeze(0).float(),
            "prob_mask": torch.from_numpy(prob_mask).unsqueeze(0).float(),
            "thresh_gt": torch.from_numpy(thresh_gt).unsqueeze(0).float(),
            "thresh_mask": torch.from_numpy(thresh_mask).unsqueeze(0).float(),
        }


def load_name_list(list_file):
    if not os.path.exists(list_file):
        return []
    with open(list_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]