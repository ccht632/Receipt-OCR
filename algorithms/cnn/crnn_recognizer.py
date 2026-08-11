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
            raise FileNotFoundError(f"CRNN weights not found: {weights_path}，Please train_crnn.py")

        checkpoint = torch.load(weights_path, map_location=self.device)
        self.char_to_idx = checkpoint["char_to_idx"]
        self.idx_to_char = checkpoint["idx_to_char"]
        self.img_height = checkpoint["img_height"]
        self.max_width = 280

        num_classes = len(self.char_to_idx)
        self.model = CRNN(num_classes=num_classes, img_height=self.img_height,
                           hidden_size=checkpoint["hidden_size"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        print(f"[CRNNRecognizer] Loaded weights: {weights_path}  Character table size: {num_classes}")

    @torch.no_grad()
    def recognize(self, cropped_image: np.ndarray) -> str:
        if cropped_image.ndim == 3:
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cropped_image

        img = resize_pad_width(gray, self.img_height, self.max_width)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device) 

        out = self.model(tensor)  
        pred_indices = out.argmax(2).squeeze(1).cpu().numpy()  
        text = ctc_greedy_decode(pred_indices, self.idx_to_char)
        return text

    @torch.no_grad()
    def recognize_batch(self, cropped_images: list) -> list:
        return [self.recognize(img) for img in cropped_images]