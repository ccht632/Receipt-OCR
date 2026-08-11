"""
CNN分支完整链路: 预处理 -> DBNet检测 -> CRNN识别 -> parser解析
统一入口给 app.py 调用:
    from pipeline_cnn import run_pipeline
    fields, debug_info = run_pipeline(image_path)
"""
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "algorithms", "cnn"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import config
from dbnet_detector import DBNetDetector
from crnn_recognizer import CRNNRecognizer
from image_processing import load_image, crop_quad
from parser import parse_receipt

_detector = None
_recognizer = None


def _get_models():
    """模型只加载一次(单例)，避免app.py每次上传图片都重新读权重。"""
    global _detector, _recognizer
    if _detector is None:
        _detector = DBNetDetector(config.DBNET_FINETUNED)
    if _recognizer is None:
        _recognizer = CRNNRecognizer(config.CRNN_WEIGHTS)
    return _detector, _recognizer


def draw_boxes(preview_image: np.ndarray, lines: list) -> np.ndarray:
    """在预处理后的图上画出检测框+识别文字，给app.py展示用。"""
    vis = preview_image.copy()
    for text, box, score in lines:
        pts = box.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, (0, 200, 0), 2)
        x, y = int(box[0][0]), max(int(box[0][1]) - 8, 12)
        cv2.putText(vis, text[:20], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return vis


def run_pipeline(image_path: str, box_thresh: float = None, score_thresh: float = None):
    """
    输入: 图片路径
    返回: (fields, debug_info)
        fields: {"company": str, "address": str, "date": str, "total": str}
        debug_info: {"preview_image", "boxes", "lines", "vis_image"} 给界面展示/调试用
    """
    detector, recognizer = _get_models()

    image = load_image(image_path)
    boxes, preview = detector.detect(image, box_thresh=box_thresh, score_thresh=score_thresh)

    lines = []
    for box, score in boxes:
        crop = crop_quad(preview, box)
        text = recognizer.recognize(crop)
        text = text.strip()
        if text:
            lines.append((text, box, score))

    # 按y坐标排序，方便调试查看时符合阅读顺序(parser内部也会自己排序，这里是为了显示)
    lines.sort(key=lambda x: (x[1][:, 1].min() + x[1][:, 1].max()) / 2)

    parser_input = [(text, box) for text, box, _ in lines]
    fields = parse_receipt(parser_input)

    debug_info = {
        "preview_image": preview,
        "boxes": boxes,
        "lines": lines,
        "vis_image": draw_boxes(preview, lines),
    }
    return fields, debug_info


if __name__ == "__main__":
    import argparse
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument("image_path")
    args = parser_arg.parse_args()

    fields, debug_info = run_pipeline(args.image_path)
    print("=== 识别出的文字行 ===")
    for text, box, score in debug_info["lines"]:
        print(f"  [{score:.2f}] {text}")
    print("\n=== 抽取字段 ===")
    for k, v in fields.items():
        print(f"  {k}: {v}")