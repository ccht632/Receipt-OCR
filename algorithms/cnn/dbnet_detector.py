"""
DBNet 推理封装。pipeline_cnn.py 只需要:
    detector = DBNetDetector(config.DBNET_FINETUNED)
    boxes = detector.detect(image)   # image: cv2读入的BGR彩色图(原图,未经预处理)
    # boxes: [(box(4,2) numpy array, score), ...]  坐标已经映射回原图尺寸
"""
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
            print(f"[DBNetDetector] 已加载权重: {weights_path}")
        else:
            print(f"[DBNetDetector] ⚠️ 找不到权重文件 {weights_path}，使用随机初始化(仅供测试代码用,不能用于实际检测)")

        self.model.eval()

    @torch.no_grad()
    def detect(self, image: np.ndarray, max_side: int = 1280,
               box_thresh: float = None, score_thresh: float = None, unclip_ratio: float = 1.5):
        """
        image: cv2读入的BGR彩色原图 (未经过任何预处理)
        返回: [(box(4,2) numpy array, score), ...]  坐标是原图尺度下的坐标
        """
        box_thresh = box_thresh if box_thresh is not None else config.DET_BOX_THRESH
        score_thresh = score_thresh if score_thresh is not None else config.DET_SCORE_THRESH

        model_input, scale, preview_image = preprocess_for_dbnet(image, max_side=max_side)
        # preview_image 是auto_crop+纠偏后的彩色图，DBNet的检测框坐标是相对这张图的，
        # 不是相对最原始输入image的(auto_crop可能改变了图像边界)。
        # pipeline_cnn.py 后续裁字用preview_image，不要用最原始的image。

        input_tensor = torch.from_numpy(model_input).unsqueeze(0).float().to(self.device)
        prob_map, thresh_map, binary_map = self.model(input_tensor)
        prob_map_np = prob_map[0, 0].cpu().numpy()

        # model_input是resize_to_multiple处理过的(缩放+右下pad到32倍数)，
        # prob_map分辨率和model_input一致，先按scale把坐标缩放回preview_image的尺度
        results = decode_boxes(prob_map_np, box_thresh=box_thresh, score_thresh=score_thresh,
                                unclip_ratio=unclip_ratio)

        boxes = []
        for box, score in results:
            box_in_preview = box / scale
            boxes.append((box_in_preview.astype(np.float32), score))

        return boxes, preview_image