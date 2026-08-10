"""
把 SROIE 的 box/ 标注 + 原图，裁剪成一张张文字行小图，
生成 CRNN 训练需要的 (裁剪图, 文字标签) 数据对。

输出：
    data/SROIE_crnn/train_crops/xxx_0001.jpg, xxx_0002.jpg, ...
    data/SROIE_crnn/train_labels.txt   每行: 相对路径\t文字内容
    data/SROIE_crnn/alphabet.txt       训练集中出现过的所有字符(CTC解码用)
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np
import config
from image_processing import load_image, crop_quad


def parse_box_file(box_path):
    """解析一个 box txt 文件，返回 [(四点坐标 np.array(4,2), transcript), ...]"""
    items = []
    with open(box_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 8)
            if len(parts) < 9:
                continue
            try:
                coords = [float(x) for x in parts[:8]]
            except ValueError:
                continue
            box = np.array(coords, dtype=np.float32).reshape(4, 2)
            transcript = parts[8].strip()
            if transcript == "" or transcript == "###":
                continue
            items.append((box, transcript))
    return items


def convert_split(img_dir, box_dir, out_crops_dir, out_labels_file, charset: set,
                   val_labels_file=None, val_ratio=0.0, seed=42):
    os.makedirs(out_crops_dir, exist_ok=True)

    if not os.path.isdir(img_dir):
        print(f"[跳过] 找不到目录: {img_dir}")
        return 0

    img_files = sorted([f for f in os.listdir(img_dir)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    # 按收据(图片)先切train/val，保证同一张收据的所有文字行只出现在一边，
    # 不会出现"同一张收据的两行,一行在train一行在val"这种数据泄漏
    val_stems = set()
    if val_ratio > 0 and val_labels_file:
        rng = random.Random(seed)
        shuffled = img_files[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val_stems = {os.path.splitext(f)[0] for f in shuffled[:n_val]}

    train_labels = []
    val_labels = []
    n_crops = 0
    n_skipped_img = 0

    for img_name in img_files:
        stem = os.path.splitext(img_name)[0]
        box_path = os.path.join(box_dir, stem + ".txt")
        if not os.path.exists(box_path):
            n_skipped_img += 1
            continue

        items = parse_box_file(box_path)
        if not items:
            continue

        try:
            image = load_image(os.path.join(img_dir, img_name))
        except ValueError:
            n_skipped_img += 1
            continue

        for idx, (box, transcript) in enumerate(items):
            # 过滤掉裁剪区域异常小的框(标注误差/噪声框)
            w = np.linalg.norm(box[0] - box[1])
            h = np.linalg.norm(box[0] - box[3])
            if w < 5 or h < 5:
                continue

            try:
                crop = crop_quad(image, box, target_height=config.CRNN_IMG_HEIGHT)
            except Exception:
                continue

            crop_name = f"{stem}_{idx:04d}.jpg"
            crop_path = os.path.join(out_crops_dir, crop_name)
            import cv2
            cv2.imwrite(crop_path, crop)

            line = f"{crop_name}\t{transcript}"
            if stem in val_stems:
                val_labels.append(line)
            else:
                train_labels.append(line)
            charset.update(list(transcript))
            n_crops += 1

    with open(out_labels_file, "w", encoding="utf-8") as f:
        f.write("\n".join(train_labels))
    if val_labels_file:
        with open(val_labels_file, "w", encoding="utf-8") as f:
            f.write("\n".join(val_labels))

    print(f"完成: {out_crops_dir}")
    if val_labels_file:
        print(f"  train: {len(train_labels)}条  val: {len(val_labels)}条  跳过的图: {n_skipped_img}")
    else:
        print(f"  裁剪文字行: {n_crops}  跳过的图(缺box/读取失败): {n_skipped_img}")
    return n_crops


def main():
    print("=" * 60)
    print("SROIE -> CRNN 数据格式转换 (裁剪文字行 + 标签)")
    print("=" * 60)

    charset = set()

    n_train = convert_split(
        config.TRAIN_IMG_DIR, config.TRAIN_BOX_DIR,
        config.CRNN_TRAIN_CROPS_DIR, config.CRNN_TRAIN_LABELS,
        charset,
        val_labels_file=config.CRNN_VAL_LABELS,
        val_ratio=config.VAL_SPLIT_RATIO,
        seed=config.VAL_SPLIT_SEED,
    )
    n_test = convert_split(
        config.TEST_IMG_DIR, config.TEST_BOX_DIR,
        config.CRNN_TEST_CROPS_DIR, config.CRNN_TEST_LABELS,
        charset,
    )

    # 生成字符表：固定顺序(排序)，第0位留给CTC的blank符号
    os.makedirs(config.CRNN_DATA_ROOT, exist_ok=True)
    alphabet = sorted(charset)
    with open(config.CRNN_ALPHABET_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(alphabet))

    if n_train == 0:
        print("\n⚠️  警告: 没有转换出任何训练样本，请检查 config.py 里的路径是否正确:")
        print(f"   TRAIN_IMG_DIR = {config.TRAIN_IMG_DIR}")
        print(f"   TRAIN_BOX_DIR = {config.TRAIN_BOX_DIR}")
    else:
        print(f"\n✅ 训练集 {n_train} 条, 测试集 {n_test} 条")
        print(f"✅ 字符表大小: {len(alphabet)} (含: {''.join(alphabet[:50])}{'...' if len(alphabet) > 50 else ''})")
        print(f"✅ 已保存到 {config.CRNN_DATA_ROOT}")


if __name__ == "__main__":
    main()