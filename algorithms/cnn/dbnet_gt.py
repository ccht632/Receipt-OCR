import numpy as np
import cv2
import pyclipper
from shapely.geometry import Polygon


def shrink_polygon(polygon: np.ndarray, ratio: float):
    poly_shape = Polygon(polygon)
    if poly_shape.area <= 0 or poly_shape.length <= 0:
        return None
    distance = poly_shape.area * (1 - ratio ** 2) / poly_shape.length
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(polygon.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    shrunk = pco.Execute(-distance)
    if not shrunk:
        return None
    return np.array(shrunk[0])


def expand_polygon(polygon: np.ndarray, ratio: float):
    poly_shape = Polygon(polygon)
    if poly_shape.area <= 0 or poly_shape.length <= 0:
        return None
    distance = poly_shape.area * (1 - ratio ** 2) / poly_shape.length
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(polygon.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = pco.Execute(distance)
    if not expanded:
        return None
    return np.array(expanded[0])


def make_shrink_map(polygons, h, w, shrink_ratio=0.4, min_text_size=8):
    prob_gt = np.zeros((h, w), dtype=np.float32)
    prob_mask = np.ones((h, w), dtype=np.float32)

    for polygon in polygons:
        height = max(polygon[:, 1]) - min(polygon[:, 1])
        width = max(polygon[:, 0]) - min(polygon[:, 0])
        if height < min_text_size or width < min_text_size:
            cv2.fillPoly(prob_mask, [polygon.astype(np.int32)], 0)
            continue

        shrunk = shrink_polygon(polygon, shrink_ratio)
        if shrunk is None or len(shrunk) < 3:
            cv2.fillPoly(prob_mask, [polygon.astype(np.int32)], 0)
            continue

        cv2.fillPoly(prob_gt, [shrunk.astype(np.int32)], 1)

    return prob_gt, prob_mask


def _point_to_segment_distance(xs, ys, p1, p2):
    sq_d1 = (xs - p1[0]) ** 2 + (ys - p1[1]) ** 2
    sq_d2 = (xs - p2[0]) ** 2 + (ys - p2[1]) ** 2
    sq_d = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    cosin = (sq_d - sq_d1 - sq_d2) / (2 * np.sqrt(sq_d1 * sq_d2) + 1e-6)
    cosin = np.clip(cosin, -1, 1)
    result = np.sqrt(sq_d1 * sq_d2 * (1 - cosin ** 2) / (sq_d + 1e-6))
    result[cosin < 0] = np.sqrt(np.minimum(sq_d1, sq_d2))[cosin < 0]
    return result


def _draw_border_map(polygon, canvas, mask, shrink_ratio=0.4):
    h, w = canvas.shape[:2]
    poly_shape = Polygon(polygon)
    if poly_shape.area <= 0:
        return
    distance = poly_shape.area * (1 - shrink_ratio ** 2) / poly_shape.length
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(polygon.tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    padded = pco.Execute(distance)
    if not padded:
        return
    padded_polygon = np.array(padded[0])
    cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

    xmin, xmax = int(padded_polygon[:, 0].min()), int(padded_polygon[:, 0].max())
    ymin, ymax = int(padded_polygon[:, 1].min()), int(padded_polygon[:, 1].max())
    xmin_c, xmax_c = max(0, xmin), min(w - 1, xmax)
    ymin_c, ymax_c = max(0, ymin), min(h - 1, ymax)
    if xmax_c < xmin_c or ymax_c < ymin_c:
        return

    width_ = xmax - xmin + 1
    height_ = ymax - ymin + 1
    poly = polygon.copy().astype(np.float64)
    poly[:, 0] -= xmin
    poly[:, 1] -= ymin

    xs = np.tile(np.linspace(0, width_ - 1, num=width_).reshape(1, width_), (height_, 1))
    ys = np.tile(np.linspace(0, height_ - 1, num=height_).reshape(height_, 1), (1, width_))

    dist_map = np.zeros((poly.shape[0], height_, width_), dtype=np.float32)
    for i in range(poly.shape[0]):
        j = (i + 1) % poly.shape[0]
        dist_map[i] = np.clip(_point_to_segment_distance(xs, ys, poly[i], poly[j]) / distance, 0, 1)
    dist_map = dist_map.min(axis=0)

    canvas_patch = canvas[ymin_c:ymax_c + 1, xmin_c:xmax_c + 1]
    dist_patch = dist_map[ymin_c - ymin:ymax_c - ymin + 1, xmin_c - xmin:xmax_c - xmin + 1]
    canvas[ymin_c:ymax_c + 1, xmin_c:xmax_c + 1] = np.fmax(1 - dist_patch, canvas_patch)


def make_threshold_map(polygons, h, w, shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7):
    canvas = np.zeros((h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.float32)
    for polygon in polygons:
        if Polygon(polygon).area <= 0:
            continue
        _draw_border_map(polygon, canvas, mask, shrink_ratio)
    thresh_gt = canvas * (thresh_max - thresh_min) + thresh_min
    return thresh_gt, mask