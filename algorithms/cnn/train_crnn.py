"""
CRNN 训练脚本，从零训练(作业要求)。
监控 val CER(字符错误率)：
  - CER改善 -> 保存best权重
  - 连续 config.CRNN_EARLY_STOP_PATIENCE 个epoch都没改善 -> 提前停止，防止过拟合
"""
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
import config
from crnn_model import CRNN
from crnn_utils import load_alphabet, ctc_greedy_decode, compute_cer
from crnn_dataset import CRNNDataset, crnn_collate_fn


class EarlyStopping:
    """监控某个指标(越小越好，比如CER)，连续patience轮没有改善就触发停止。"""

    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, score):
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
            return True  # 表示这一轮是新的最优
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_train_epoch(model, loader, ctc_loss, optimizer, device):
    model.train()
    total_loss = 0.0
    n = 0
    for images, targets, target_lengths, _ in loader:
        images, targets, target_lengths = images.to(device), targets.to(device), target_lengths.to(device)

        optimizer.zero_grad()
        out = model(images)
        input_lengths = torch.full((images.size(0),), out.size(0), dtype=torch.long).to(device)
        loss = ctc_loss(out, targets, input_lengths, target_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # CTC训练容易梯度爆炸,裁剪一下
        optimizer.step()

        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def run_val_epoch(model, loader, ctc_loss, device, idx_to_char):
    """返回 (val_loss, val_cer)"""
    model.eval()
    total_loss = 0.0
    total_cer = 0.0
    n_batches = 0
    n_samples = 0

    for images, targets, target_lengths, texts in loader:
        images_d = images.to(device)
        targets_d, target_lengths_d = targets.to(device), target_lengths.to(device)

        out = model(images_d)  # (W', B, C)
        input_lengths = torch.full((images.size(0),), out.size(0), dtype=torch.long).to(device)
        loss = ctc_loss(out, targets_d, input_lengths, target_lengths_d)
        total_loss += loss.item()
        n_batches += 1

        pred_indices = out.argmax(2).permute(1, 0).cpu().numpy()  # (B, W')
        for i, text in enumerate(texts):
            pred_text = ctc_greedy_decode(pred_indices[i], idx_to_char)
            total_cer += compute_cer(pred_text, text)
            n_samples += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_cer = total_cer / max(n_samples, 1)
    return avg_loss, avg_cer


def main():
    device = get_device()
    print(f"使用设备: {device}")
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.RUNS_DIR, exist_ok=True)

    if not os.path.exists(config.CRNN_ALPHABET_FILE):
        print("⚠️ 找不到 alphabet.txt，请先跑 prepare_crnn_data.py")
        return
    char_to_idx, idx_to_char = load_alphabet(config.CRNN_ALPHABET_FILE)
    num_classes = len(char_to_idx)
    print(f"字符表大小: {num_classes}")

    train_ds = CRNNDataset(config.CRNN_TRAIN_CROPS_DIR, config.CRNN_TRAIN_LABELS, char_to_idx,
                            img_height=config.CRNN_IMG_HEIGHT, max_width=config.CRNN_IMG_MAX_WIDTH)
    val_ds = CRNNDataset(config.CRNN_TRAIN_CROPS_DIR, config.CRNN_VAL_LABELS, char_to_idx,
                          img_height=config.CRNN_IMG_HEIGHT, max_width=config.CRNN_IMG_MAX_WIDTH)
    print(f"train: {len(train_ds)}条  val: {len(val_ds)}条")

    if len(train_ds) == 0:
        print("⚠️ 没有训练数据")
        return

    train_loader = DataLoader(train_ds, batch_size=config.CRNN_BATCH_SIZE, shuffle=True,
                               collate_fn=crnn_collate_fn, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.CRNN_BATCH_SIZE, shuffle=False,
                             collate_fn=crnn_collate_fn, num_workers=2)

    model = CRNN(num_classes=num_classes, img_height=config.CRNN_IMG_HEIGHT,
                 hidden_size=config.CRNN_HIDDEN_SIZE).to(device)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.CRNN_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    early_stopping = EarlyStopping(patience=config.CRNN_EARLY_STOP_PATIENCE)

    log_path = os.path.join(config.RUNS_DIR, "crnn_train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,val_cer,lr,seconds\n")

    for epoch in range(1, config.CRNN_EPOCHS + 1):
        t0 = time.time()
        train_loss = run_train_epoch(model, train_loader, ctc_loss, optimizer, device)

        if len(val_ds) > 0:
            val_loss, val_cer = run_val_epoch(model, val_loader, ctc_loss, device, idx_to_char)
        else:
            val_loss, val_cer = float("nan"), float("nan")

        scheduler.step(val_loss if len(val_ds) > 0 else train_loss)
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"[Epoch {epoch}/{config.CRNN_EPOCHS}] "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_CER={val_cer:.4f}  "
              f"lr={lr_now:.2e}  time={elapsed:.1f}s")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_cer:.6f},{lr_now:.6e},{elapsed:.1f}\n")

        if len(val_ds) > 0:
            is_best = early_stopping.step(val_cer)
            if is_best:
                torch.save({
                    "model_state": model.state_dict(),
                    "char_to_idx": char_to_idx,
                    "idx_to_char": idx_to_char,
                    "img_height": config.CRNN_IMG_HEIGHT,
                    "hidden_size": config.CRNN_HIDDEN_SIZE,
                }, config.CRNN_WEIGHTS)
                print(f"  ✅ val_CER改善至{val_cer:.4f}，保存best模型到 {config.CRNN_WEIGHTS}")

            if early_stopping.should_stop:
                print(f"\n🛑 Early stopping：val_CER连续{config.CRNN_EARLY_STOP_PATIENCE}轮没有改善，"
                      f"在第{epoch}轮停止训练，防止过拟合。最佳val_CER={early_stopping.best_score:.4f}")
                break
        else:
            torch.save({
                "model_state": model.state_dict(),
                "char_to_idx": char_to_idx,
                "idx_to_char": idx_to_char,
                "img_height": config.CRNN_IMG_HEIGHT,
                "hidden_size": config.CRNN_HIDDEN_SIZE,
            }, config.CRNN_WEIGHTS)

    print(f"\n训练结束。日志: {log_path}")


if __name__ == "__main__":
    main()