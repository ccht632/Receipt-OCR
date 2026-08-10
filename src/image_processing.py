"""
共用图像预处理模块 (CNN分支 与 YOLO分支 共用)

合并说明：
- 前半部分(灰度判断/自动裁剪收据边界/亮度对比度判断/去噪/纠偏)完全共用，
  两条pipeline调用同一套逻辑，保证预处理效果一致。
  这部分函数都改成了"通道无关"：传入灰度图或彩色图都能正常工作，
  内部需要用到灰度信息的地方(边缘检测/二值化)会自动转换，
  但几何变换(裁剪/旋转)会应用在传入的原图上，不会强制丢失颜色。

- 后半部分是两个"分支适配器"，只在这里才分岔：
    preprocess_image()     -> YOLO分支入口(队友原有函数名，不改，保证队友的pipeline_yolo.py不用改代码)
                               灰度 + 正方形letterbox + 归一化
    preprocess_for_dbnet() -> CNN分支入口(新增)
                               保留3通道彩色 + 32倍数缩放 + ImageNet均值方差归一化
                               (因为DBNet用的预训练权重是ImageNet上训练的3通道backbone，
                                转灰度会丢失预训练权重能利用的颜色统计特征)

- crop_quad() 是CNN分支专属：DBNet检测出文字框后，裁剪出文字行小图给CRNN识别用。
"""
import cv2
import numpy as np

YOLO_INPUT_SIZE = 640
DBNET_MULTIPLE = 32          # DBNet backbone下采样5次(2^5=32)，输入边长必须是32的倍数
NOISE_KERNEL = 3

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================
# 读图
# ============================================================
def load_image(path: str) -> np.ndarray:
    """从路径读图，返回 BGR 彩色 numpy array。用 imdecode 避免中文路径读取失败的坑。"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {path}")
    return img


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    """内部工具：不管传入灰度图还是彩色图，都返回灰度版本，用于需要单通道的算法(边缘检测/二值化等)。"""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    return _ensure_gray(image)


# ============================================================
# 自动判断(通道无关：内部统一转灰度算统计量)
# ============================================================
def is_dark(image, threshold=100):
    return np.mean(_ensure_gray(image)) < threshold


def is_low_contrast(image, threshold=40):
    return _ensure_gray(image).std() < threshold


def is_noisy(image, threshold=15):
    gray = _ensure_gray(image)
    denoised = cv2.medianBlur(gray, 3)
    diff = cv2.absdiff(gray, denoised)
    return np.mean(diff) > threshold


def is_blurry(image, threshold=100):
    return cv2.Laplacian(_ensure_gray(image), cv2.CV_64F).var() < threshold


def is_skewed(image, threshold=3):
    return abs(get_skew_angle(image)) > threshold


def get_skew_angle(image):
    gray = _ensure_gray(image)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        return 0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    return angle


# ============================================================
# 增强 / 去噪 / 纠偏 (通道无关：接受灰度或彩色，返回同样通道数)
# ============================================================
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    if image.ndim == 2:
        return clahe.apply(image)
    # 彩色图：只在LAB的L通道(亮度)上做CLAHE，不破坏颜色信息，这点是CNN分支需要的
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def remove_noise(image):
    return cv2.medianBlur(image, NOISE_KERNEL)


def deskew(image):
    angle = get_skew_angle(image)
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)


# ============================================================
# 收据边界检测 + 透视裁剪 (队友原版逻辑，改成通道无关)
# ============================================================
def auto_crop_receipt(image):
    gray = _ensure_gray(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    candidates = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    receipt_corners = None
    for c in candidates:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.15 * gray.shape[0] * gray.shape[1]:
            receipt_corners = approx
            break

    if receipt_corners is None and candidates:
        largest = candidates[0]
        if cv2.contourArea(largest) > 0.15 * gray.shape[0] * gray.shape[1]:
            rect = cv2.minAreaRect(largest)
            receipt_corners = cv2.boxPoints(rect).astype(np.int32).reshape(4, 1, 2)

    if receipt_corners is None:
        return image

    # 用原图(灰度或彩色都行)做透视变换，不强制转灰度，保留调用方传入的通道数
    return warp_to_rectangle(image, receipt_corners.reshape(4, 2))


def warp_to_rectangle(image, pts):
    rect = order_corners(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    dst = np.array([
        [0, 0], [max_width - 1, 0],
        [max_width - 1, max_height - 1], [0, max_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect.astype("float32"), dst)
    return cv2.warpPerspective(image, matrix, (max_width, max_height))


def order_corners(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# ============================================================
# 共用主流程
# ============================================================
def preprocess_common(image, grayscale: bool):
    """
    两条pipeline的共用入口。
    grayscale=True  -> 输出单通道灰度图 (YOLO分支用)
    grayscale=False -> 输出3通道彩色图 (DBNet分支用，保留颜色给预训练backbone用)
    裁剪/增强/纠偏的判断逻辑完全一致，只是最后携带的通道数不同。
    """
    if grayscale:
        image = to_grayscale(image)

    image = auto_crop_receipt(image)

    if is_dark(image) or is_low_contrast(image):
        image = apply_clahe(image)

    if is_skewed(image):
        image = deskew(image)

    preview_image = image.copy()
    return image, preview_image


# ============================================================
# YOLO 分支入口 (函数名保持队友原来的 preprocess_image，不用改队友的调用代码)
# ============================================================
def letterbox_resize(image, target_size=640, pad_color=255):
    h, w = image.shape
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.full((target_size, target_size), pad_color, dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def preprocess_image(image):
    """YOLO分支预处理入口(原队友函数，逻辑不变，只是底层函数改成了共用版本)。"""
    image = np.array(image)
    gray, preview_image = preprocess_common(image, grayscale=True)

    if is_noisy(gray) and not is_blurry(gray):
        yolo_input = remove_noise(gray)
    else:
        yolo_input = gray

    yolo_input = letterbox_resize(yolo_input, YOLO_INPUT_SIZE)
    yolo_input = yolo_input.astype(np.float32) / 255.0
    return yolo_input, preview_image


# ============================================================
# DBNet 分支入口 (新增)
# ============================================================
def resize_to_multiple(image, max_side=1280, multiple=DBNET_MULTIPLE):
    """
    等比例缩放到最长边<=max_side，再向右/下pad到32的倍数(DBNet网络结构要求)。
    返回 (处理后的图, scale)，scale 用于把检测框坐标映射回原图尺寸(pad部分不需要映射)。
    """
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        new_h, new_w = int(h * scale), int(w * scale)
    else:
        new_h, new_w = h, w
        scale = 1.0

    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    pad_h = (multiple - new_h % multiple) % multiple
    pad_w = (multiple - new_w % multiple) % multiple
    padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded, scale


def normalize_imagenet(image_bgr):
    """BGR彩色图 -> RGB -> [0,1] -> ImageNet均值方差归一化 -> CHW(pytorch输入格式)。"""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return normalized.transpose(2, 0, 1)  # HWC -> CHW


def preprocess_for_dbnet(image, max_side=1280):
    """
    DBNet分支预处理入口。
    返回 (归一化后的CHW数组给模型推理用, scale, preview_image给界面展示用)
    """
    image = np.array(image)
    color, preview_image = preprocess_common(image, grayscale=False)

    if is_noisy(color) and not is_blurry(color):
        color = remove_noise(color)

    resized, scale = resize_to_multiple(color, max_side)
    model_input = normalize_imagenet(resized)
    return model_input, scale, preview_image


# ============================================================
# CNN分支专属：裁剪文字行给CRNN识别用
# ============================================================
def crop_quad(image: np.ndarray, box: np.ndarray, target_height: int = 32) -> np.ndarray:
    """
    按四点坐标(box, shape=(4,2))做透视变换，裁出文字行矩形图，
    并统一缩放到 target_height (CRNN 输入要求固定高度)。
    box 顺序: [左上, 右上, 右下, 左下]
    """
    box = box.astype(np.float32)
    w1 = np.linalg.norm(box[0] - box[1])
    w2 = np.linalg.norm(box[3] - box[2])
    h1 = np.linalg.norm(box[0] - box[3])
    h2 = np.linalg.norm(box[1] - box[2])
    dst_w = max(int(max(w1, w2)), 1)
    dst_h = max(int(max(h1, h2)), 1)

    dst_pts = np.array([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]],
                        dtype=np.float32)
    M = cv2.getPerspectiveTransform(box, dst_pts)
    warped = cv2.warpPerspective(image, M, (dst_w, dst_h))

    scale = target_height / warped.shape[0]
    new_w = max(int(warped.shape[1] * scale), 1)
    warped = cv2.resize(warped, (new_w, target_height), interpolation=cv2.INTER_CUBIC)
    return warped