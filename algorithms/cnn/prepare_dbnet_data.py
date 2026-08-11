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
        print(f"[Skip] Directory not found: {img_dir}")
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

        gt_lines = []
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
                transcript = parts[8]
                gt_lines.append(",".join(parts[:8]) + "," + transcript)

        if not gt_lines:
            skipped += 1
            continue

        src_img_path = os.path.join(img_dir, img_name)
        dst_img_path = os.path.join(out_img_dir, img_name)
        shutil.copyfile(src_img_path, dst_img_path)

        dst_gt_path = os.path.join(out_gt_dir, stem + ".txt")
        with open(dst_gt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(gt_lines))

        valid_names.append(img_name)

    if val_ratio > 0 and val_list_file:
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

        print(f"Complete: {out_img_dir}")
        print(f"  train: {len(train_names)}  val: {len(val_names)}  Skip: {skipped}")
        return len(train_names)
    else:
        with open(out_list_file, "w", encoding="utf-8") as f:
            f.write("\n".join(valid_names))
        print(f"Complete: {out_img_dir}")
        print(f"  Valid samples: {len(valid_names)}  Skip (Missing box/Parsing failed): {skipped}")
        return len(valid_names)


def main():
    print("=" * 60)
    print("SROIE -> DBNet Data Format Conversion")
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
        print("\n⚠️  Warning: No training samples were generated. Please check if the paths in config.py are correct.:")
        print(f"   TRAIN_IMG_DIR = {config.TRAIN_IMG_DIR}")
        print(f"   TRAIN_BOX_DIR = {config.TRAIN_BOX_DIR}")
    else:
        print(f"\nTrain Set {n_train} , Test Set {n_test} has been generated {config.DBNET_DATA_ROOT}")


if __name__ == "__main__":
    main()