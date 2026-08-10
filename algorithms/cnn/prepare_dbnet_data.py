"""
把 SROIE 的 box/ 标注转换成 DBNet 训练所需的数据格式。

SROIE box 文件格式(每行)：
    x1,y1,x2,y2,x3,y3,x4,y4,transcript
这恰好和 DBNet 官方实现(MhLiao/DB)用的 ICDAR 格式一致，
所以这里主要做的是：
    1. 复制图片到 DBNET_TRAIN_IMG_DIR / DBNET_TEST_IMG_DIR
    2. 把 box 文件复制成对应的 gt 文件(文件名对齐: xxx.jpg <-> xxx.txt)
    3. 生成 train_list.txt / test_list.txt (每行一个图片文件名，训练脚本读取用)

注意：transcript 里如果本身含有逗号(比如 "TOTAL,80" 这种异常数据)，
SROIE 原数据里较少见，但保险起见用 maxsplit=8 只切前8个坐标，
剩下全部当成 transcript，避免把文字内容错切。
"""
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))
import config


def convert_split(img_dir, box_dir, out_img_dir, out_gt_dir, out_list_file,
                   val_list_file=None, val_ratio=0.0, seed=42):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_gt_dir, exist_ok=True)

    if not os.path.isdir(img_dir):
        print(f"[跳过] 找不到目录: {img_dir}")
        return 0

    img_files = sorted([f for f in os.listdir(img_dir)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    valid_names = []
    skipped = 0

    for img_name in img_files:
        stem = os.path.splitext(img_name)[0]
        box_path = os.path.join(box_dir, stem + ".txt")

        if not os.path.exists(box_path):
            skipped += 1
            continue

        # --- 读取并校验 box 文件，过滤掉解析失败的行 ---
        gt_lines = []
        with open(box_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",", 8)  # 前8个是坐标，第9个开始是transcript
                if len(parts) < 9:
                    continue  # 格式不对的行跳过
                try:
                    coords = [float(x) for x in parts[:8]]
                except ValueError:
                    continue
                transcript = parts[8]
                # DBNet官方格式里用 "###" 标记不参与loss计算的困难样本，
                # SROIE本身文字都比较清楚，这里不特殊处理，全部当有效样本
                gt_lines.append(",".join(parts[:8]) + "," + transcript)

        if not gt_lines:
            skipped += 1
            continue

        # --- 复制图片 ---
        src_img_path = os.path.join(img_dir, img_name)
        dst_img_path = os.path.join(out_img_dir, img_name)
        shutil.copyfile(src_img_path, dst_img_path)

        # --- 写 gt 文件 ---
        dst_gt_path = os.path.join(out_gt_dir, stem + ".txt")
        with open(dst_gt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(gt_lines))

        valid_names.append(img_name)

    if val_ratio > 0 and val_list_file:
        # 按图片(收据)切分train/val，不能按box行切，否则同一张收据的行
        # 同时出现在train和val里，会造成数据泄漏
        rng = random.Random(seed)
        shuffled = valid_names[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val_names = sorted(shuffled[:n_val])
        train_names = sorted(shuffled[n_val:])

        with open(out_list_file, "w", encoding="utf-8") as f:
            f.write("\n".join(train_names))
        with open(val_list_file, "w", encoding="utf-8") as f:
            f.write("\n".join(val_names))

        print(f"完成: {out_img_dir}")
        print(f"  train: {len(train_names)}  val: {len(val_names)}  跳过: {skipped}")
        return len(train_names)
    else:
        with open(out_list_file, "w", encoding="utf-8") as f:
            f.write("\n".join(valid_names))
        print(f"完成: {out_img_dir}")
        print(f"  有效样本: {len(valid_names)}  跳过(缺box/解析失败): {skipped}")
        return len(valid_names)


def main():
    print("=" * 60)
    print("SROIE -> DBNet 数据格式转换")
    print("=" * 60)

    n_train = convert_split(
        config.TRAIN_IMG_DIR, config.TRAIN_BOX_DIR,
        config.DBNET_TRAIN_IMG_DIR, config.DBNET_TRAIN_GT_DIR,
        config.DBNET_TRAIN_LIST,
        val_list_file=config.DBNET_VAL_LIST,
        val_ratio=config.VAL_SPLIT_RATIO,
        seed=config.VAL_SPLIT_SEED,
    )
    n_test = convert_split(
        config.TEST_IMG_DIR, config.TEST_BOX_DIR,
        config.DBNET_TEST_IMG_DIR, config.DBNET_TEST_GT_DIR,
        config.DBNET_TEST_LIST,
    )

    if n_train == 0:
        print("\n⚠️  警告: 没有转换出任何训练样本，请检查 config.py 里的路径是否正确:")
        print(f"   TRAIN_IMG_DIR = {config.TRAIN_IMG_DIR}")
        print(f"   TRAIN_BOX_DIR = {config.TRAIN_BOX_DIR}")
    else:
        print(f"\n✅ 训练集 {n_train} 张, 测试集 {n_test} 张已生成到 {config.DBNET_DATA_ROOT}")


if __name__ == "__main__":
    main()