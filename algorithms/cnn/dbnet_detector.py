import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import config
from dbnet_model import DBNet
from dbnet_postprocess import decode_boxes
from image_processing import preprocess_for_dbnet


class DBNetDetector:
    def __init__(self, weights_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DBNet(backbone=config.DBNET_BACKBONE, pretrained=False).to(self.device)

        weights_path = weights_path or config.DBNET_FINETUNED
        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"[DBNetDetector] Weights loaded: {weights_path}")
        else:
            print(f"[DBNetDetector] Weight file not found {weights_path}，Use random initialization (for test code only, not for actual testing).")

        self.model.eval()

    @torch.no_grad()
    def detect(self, image: np.ndarray, max_side: int = 1280,
               box_thresh: float = None, score_thresh: float = None, unclip_ratio: float = 1.5):
        box_thresh = box_thresh if box_thresh is not None else config.DET_BOX_THRESH
        score_thresh = score_thresh if score_thresh is not None else config.DET_SCORE_THRESH

        model_input, scale, preview_image = preprocess_for_dbnet(image, max_side=max_side)
        input_tensor = torch.from_numpy(model_input).unsqueeze(0).float().to(self.device)
        prob_map, thresh_map, binary_map = self.model(input_tensor)
        prob_map_np = prob_map[0, 0].cpu().numpy()

        results = decode_boxes(prob_map_np, box_thresh=box_thresh, score_thresh=score_thresh,
                                unclip_ratio=unclip_ratio)

        boxes = []
        for box, score in results:
            box_in_preview = box / scale
            boxes.append((box_in_preview.astype(np.float32), score))

        return boxes, preview_image