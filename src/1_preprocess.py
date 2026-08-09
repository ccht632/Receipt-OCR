import cv2
import numpy as np

YOLO_INPUT_SIZE = 640  
NOISE_KERNEL = 3


def preprocess_image(image):

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = auto_crop_receipt(image)

    # Enhance contrast only when needed (dark OR low-contrast/washed-out)
    if is_dark(image) or is_low_contrast(image):
        image = apply_clahe(image)

    if is_skewed(image):
        image = deskew(image)

    preview_image = image.copy()

    # Denoise only if the image is actually noisy, and only for the YOLO branch
    if is_noisy(image) and not is_blurry(image):
        yolo_input = remove_noise(image)
    else:
        yolo_input = image

    # YOLO input — letterbox resize keeps aspect ratio so text isn't stretched/squashed
    yolo_input = letterbox_resize(yolo_input, YOLO_INPUT_SIZE)
    yolo_input = yolo_input.astype(np.float32) / 255.0

    return yolo_input, preview_image


def letterbox_resize(image, target_size=640, pad_color=255):
    h, w = image.shape
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.full((target_size, target_size), pad_color, dtype=np.uint8)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized

    return canvas


def is_dark(image):
    brightness = np.mean(image)
    return brightness < 100


def is_low_contrast(image, threshold=40):
    return image.std() < threshold


def is_noisy(image, threshold=15):
    denoised = cv2.medianBlur(image, 3)
    diff = cv2.absdiff(image, denoised)
    return np.mean(diff) > threshold


def is_blurry(image, threshold=100):
    return cv2.Laplacian(image, cv2.CV_64F).var() < threshold


def is_skewed(image):
    angle = get_skew_angle(image)
    return abs(angle) > 3


def apply_clahe(image):
    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8)
    )
    return clahe.apply(image)


def remove_noise(image):
    return cv2.medianBlur(image, NOISE_KERNEL)


def auto_crop_receipt(image):

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return image

    candidates = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
 
    receipt_corners = None
    for c in candidates:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.15 * image.shape[0] * image.shape[1]:
            receipt_corners = approx
            break

    if receipt_corners is None:
        if candidates:
            largest = candidates[0]
            if cv2.contourArea(largest) > 0.15 * image.shape[0] * image.shape[1]:
                rect = cv2.minAreaRect(largest)
                receipt_corners = cv2.boxPoints(rect).astype(np.int32).reshape(4, 1, 2)

    if receipt_corners is None:
        return image

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
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
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


def get_skew_angle(image):

    _, thresh = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    coords = np.column_stack(np.where(thresh > 0))

    if len(coords) == 0:
        return 0

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle += 90

    return angle


def deskew(image):

    angle = get_skew_angle(image)

    h, w = image.shape

    matrix = cv2.getRotationMatrix2D(
        (w / 2, h / 2),
        angle,
        1.0
    )

    return cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )