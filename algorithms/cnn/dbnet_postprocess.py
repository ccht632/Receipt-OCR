"""
DBNet 后处理: 把模型输出的概率图转换成文字框坐标列表。
流程: 二值化 -> 找轮廓 -> 用轮廓内平均概率算分数(过滤低分) -> unclip往外扩(训练时shrink过,
      这里要还原回接近原始文字框大小) -> 转成4点矩形框。
"""
import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon


def unclip(box: np.ndarray, ratio: float = 1.5):
    """把shrink过的框往外扩回接近原始大小，ratio是扩张系数(与训练时shrink_ratio对应,经验值1.5)。"""
    poly = Polygon(box)
    if poly.area <= 0 or poly.length <= 0:
        return None
    distance = poly.area * ratio / poly.length
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(box.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = pco.Execute(distance)
    if not expanded:
        return None
    return np.array(expanded[0])


def box_score(prob_map: np.ndarray, box: np.ndarray) -> float:
    """轮廓内的平均概率值，作为该框的置信度分数。"""
    h, w = prob_map.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    box_int = box.astype(np.int32)
    cv2.fillPoly(mask, [box_int], 1)
    if mask.sum() == 0:
        return 0.0
    return float(prob_map[mask == 1].mean())


def polygon_to_rect(contour: np.ndarray):
    """轮廓 -> 最小外接矩形的4个角点，按 [左上,右上,右下,左下] 排序。"""
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    # 按x+y排左上, x-y排右上, 保证顺序一致方便后续crop_quad使用
    s = box.sum(axis=1)
    diff = np.diff(box, axis=1).flatten()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = box[np.argmin(s)]
    ordered[2] = box[np.argmax(s)]
    ordered[1] = box[np.argmin(diff)]
    ordered[3] = box[np.argmax(diff)]
    return ordered


def decode_boxes(prob_map: np.ndarray, box_thresh: float = 0.5, score_thresh: float = 0.5,
                  unclip_ratio: float = 1.5, min_size: int = 5, max_boxes: int = 500):
    """
    prob_map: 2D numpy array, 模型输出的概率图 (H, W), 值域[0,1]
    返回: [(box(4,2) numpy array, score), ...]，box坐标是在prob_map所在的分辨率下的坐标
    """
    binary = (prob_map > box_thresh).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for contour in contours[:max_boxes]:
        if cv2.contourArea(contour) < min_size * min_size:
            continue

        rect_box = polygon_to_rect(contour.reshape(-1, 2))
        score = box_score(prob_map, rect_box)
        if score < score_thresh:
            continue

        expanded = unclip(rect_box, unclip_ratio)
        if expanded is None or len(expanded) < 4:
            continue

        final_box = polygon_to_rect(expanded)

        w = np.linalg.norm(final_box[0] - final_box[1])
        h = np.linalg.norm(final_box[0] - final_box[3])
        if w < min_size or h < min_size:
            continue

        results.append((final_box, score))

    return results