"""
DBNet 训练脚本。
从 ImageNet 预训练权重开始 fine-tune，每个epoch记录 train loss 和 val loss，
val loss 更低时保存 best checkpoint 到 config.DBNET_FINETUNED。
"""
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
import config
from dbnet_model import DBNet
from dbnet_loss import DBLoss
from dbnet_dataset import DBNetDataset, load_name_list


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """optimizer给了就是训练模式(会更新参数), 不给就是验证模式(只算loss不更新)。"""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_parts = {"loss_prob": 0.0, "loss_thresh": 0.0, "loss_binary": 0.0}
    n_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            if is_train:
                optimizer.zero_grad()
                prob, thresh, binary = model(batch["image"])
                loss, parts = loss_fn(prob, thresh, binary, batch)
                loss.backward()
                optimizer.step()
            else:
                prob, thresh, binary = model(batch["image"])
                loss, parts = loss_fn(prob, thresh, binary, batch)

            total_loss += loss.item()
            for k in total_parts:
                total_parts[k] += parts[k]
            n_batches += 1

    n_batches = max(n_batches, 1)
    avg_loss = total_loss / n_batches
    avg_parts = {k: v / n_batches for k, v in total_parts.items()}
    return avg_loss, avg_parts


def main():
    device = get_device()
    print(f"使用设备: {device}")
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.RUNS_DIR, exist_ok=True)

    train_names = load_name_list(config.DBNET_TRAIN_LIST)
    val_names = load_name_list(config.DBNET_VAL_LIST)
    print(f"train: {len(train_names)}张  val: {len(val_names)}张")

    if len(train_names) == 0:
        print("⚠️ 没有训练数据，请先跑 prepare_dbnet_data.py")
        return

    train_ds = DBNetDataset(config.DBNET_TRAIN_IMG_DIR, config.DBNET_TRAIN_GT_DIR,
                             train_names, input_size=config.DBNET_INPUT_SIZE)
    val_ds = DBNetDataset(config.DBNET_TRAIN_IMG_DIR, config.DBNET_TRAIN_GT_DIR,
                           val_names, input_size=config.DBNET_INPUT_SIZE)

    train_loader = DataLoader(train_ds, batch_size=config.DBNET_BATCH_SIZE,
                               shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.DBNET_BATCH_SIZE,
                             shuffle=False, num_workers=2)

    model = DBNet(backbone=config.DBNET_BACKBONE, pretrained=True).to(device)
    loss_fn = DBLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.DBNET_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.DBNET_EPOCHS)

    log_path = os.path.join(config.RUNS_DIR, "dbnet_train_log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,val_loss,train_prob,train_thresh,train_binary,"
                "val_prob,val_thresh,val_binary,lr,seconds\n")

    best_val_loss = float("inf")
    for epoch in range(1, config.DBNET_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_parts = run_epoch(model, train_loader, loss_fn, device, optimizer)

        if len(val_ds) > 0:
            val_loss, val_parts = run_epoch(model, val_loader, loss_fn, device, optimizer=None)
        else:
            val_loss, val_parts = float("nan"), {"loss_prob": 0, "loss_thresh": 0, "loss_binary": 0}

        scheduler.step()
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(f"[Epoch {epoch}/{config.DBNET_EPOCHS}] "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"lr={lr_now:.2e}  time={elapsed:.1f}s")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},"
                    f"{train_parts['loss_prob']:.6f},{train_parts['loss_thresh']:.6f},{train_parts['loss_binary']:.6f},"
                    f"{val_parts['loss_prob']:.6f},{val_parts['loss_thresh']:.6f},{val_parts['loss_binary']:.6f},"
                    f"{lr_now:.6e},{elapsed:.1f}\n")

        if len(val_ds) > 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.DBNET_FINETUNED)
            print(f"  ✅ val_loss改善，保存best模型到 {config.DBNET_FINETUNED}")
        elif len(val_ds) == 0:
            torch.save(model.state_dict(), config.DBNET_FINETUNED)

    print(f"\n训练完成。最佳 val_loss = {best_val_loss:.4f}")
    print(f"训练日志: {log_path}")


if __name__ == "__main__":
    main()