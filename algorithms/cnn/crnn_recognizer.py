"""
CRNN 推理封装。pipeline_cnn.py 用法:
    recognizer = CRNNRecognizer(config.CRNN_WEIGHTS)
    text = recognizer.recognize(cropped_line_image)   # cropped_line_image: crop_quad()裁出来的图(灰度或彩色都行)
"""
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from crnn_model import CRNN
from crnn_utils import ctc_greedy_decode
from crnn_dataset import resize_pad_width


class CRNNRecognizer:
    def __init__(self, weights_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"找不到CRNN权重: {weights_path}，请先跑 train_crnn.py")

        checkpoint = torch.load(weights_path, map_location=self.device)
        self.char_to_idx = checkpoint["char_to_idx"]
        self.idx_to_char = checkpoint["idx_to_char"]
        self.img_height = checkpoint["img_height"]
        self.max_width = 280  # 和训练时config.CRNN_IMG_MAX_WIDTH保持一致

        num_classes = len(self.char_to_idx)
        self.model = CRNN(num_classes=num_classes, img_height=self.img_height,
                           hidden_size=checkpoint["hidden_size"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        print(f"[CRNNRecognizer] 已加载权重: {weights_path}  字符表大小: {num_classes}")

    @torch.no_grad()
    def recognize(self, cropped_image: np.ndarray) -> str:
        """cropped_image: crop_quad()裁出来的文字行图(BGR彩色或灰度都可)。返回识别出的文字。"""
        if cropped_image.ndim == 3:
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cropped_image

        img = resize_pad_width(gray, self.img_height, self.max_width)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,H,W)

        out = self.model(tensor)  # (W', 1, num_classes+1)
        pred_indices = out.argmax(2).squeeze(1).cpu().numpy()  # (W',)
        text = ctc_greedy_decode(pred_indices, self.idx_to_char)
        return text

    @torch.no_grad()
    def recognize_batch(self, cropped_images: list) -> list:
        """批量识别，提升速度用。返回文字列表，顺序和输入一致。"""
        return [self.recognize(img) for img in cropped_images]