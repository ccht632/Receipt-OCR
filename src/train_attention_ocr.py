"""
train_attention_ocr.py (改进版)

训练一个 Attention-based 文字识别模型，基于 SROIE2019 数据集。

本版改进：
    1. 加入训练集/验证集拆分，能实际看到有没有过拟合
    2. 加入 Dropout，降低过拟合概率
    3. 加入 Learning Rate Scheduler，根据验证集表现自动调整学习率
    4. 加入 Early Stopping，验证集效果连续几轮不提升就自动停止，避免浪费时间训练已经过拟合的模型

运行:
    python src/train_attention_ocr.py
"""

import os
import glob
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import numpy as np
import time
from tqdm import tqdm

# ---------------- 配置区 ----------------
DATA_ROOT = "data/SROIE2019/train"
IMG_DIR = os.path.join(DATA_ROOT, "img")
BOX_DIR = os.path.join(DATA_ROOT, "box")

IMG_HEIGHT = 32
IMG_MAX_WIDTH = 200
MAX_TEXT_LEN = 32
BATCH_SIZE = 32
EPOCHS = 50                    # 加大上限，反正有Early Stopping会自动提前结束
LR = 0.001
EMBED_DIM = 256
HIDDEN_DIM = 256
DROPOUT = 0.3                  # 新增：Dropout比例
VAL_SPLIT = 0.15               # 新增：15%数据作为验证集
EARLY_STOP_PATIENCE = 5        # 新增：验证集连续5轮没进步就停止
MODEL_SAVE_PATH = "models/ocr/attention_ocr_model.pth"
# -----------------------------------------

CHARSET = string.digits + string.ascii_uppercase + string.ascii_lowercase + " .,:-/()&%$"
CHAR2IDX = {c: i + 3 for i, c in enumerate(CHARSET)}
IDX2CHAR = {i + 3: c for i, c in enumerate(CHARSET)}
PAD, SOS, EOS = 0, 1, 2
VOCAB_SIZE = len(CHARSET) + 3


def encode_text(text):
    ids = [SOS] + [CHAR2IDX[c] for c in text if c in CHAR2IDX][:MAX_TEXT_LEN - 2] + [EOS]
    ids += [PAD] * (MAX_TEXT_LEN - len(ids))
    return ids[:MAX_TEXT_LEN]


def decode_prediction(indices):
    chars = []
    for idx in indices:
        if idx == EOS:
            break
        if idx in IDX2CHAR:
            chars.append(IDX2CHAR[idx])
    return "".join(chars)


# ---------------- 数据集 ----------------

class SROIELineDataset(Dataset):
    def __init__(self, img_dir, box_dir):
        self.samples = []

        box_files = glob.glob(os.path.join(box_dir, "*.txt"))
        for box_file in box_files:
            basename = os.path.splitext(os.path.basename(box_file))[0]
            img_path = os.path.join(img_dir, basename + ".jpg")
            if not os.path.exists(img_path):
                img_path = os.path.join(img_dir, basename + ".png")
            if not os.path.exists(img_path):
                continue

            with open(box_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split(",", 8)
                    if len(parts) < 9:
                        continue
                    try:
                        coords = list(map(int, parts[:8]))
                    except ValueError:
                        continue
                    text = parts[8].strip()
                    if not text:
                        continue
                    self.samples.append((img_path, coords, text))

        print(f"Loaded {len(self.samples)} text-line samples from SROIE2019")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, coords, text = self.samples[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        x1, y1, x2, y2, x3, y3, x4, y4 = coords
        xs, ys = [x1, x2, x3, x4], [y1, y2, y3, y4]
        left, right = max(0, min(xs)), max(xs)
        top, bottom = max(0, min(ys)), max(ys)

        crop = image[top:bottom, left:right]
        if crop.size == 0:
            crop = np.zeros((IMG_HEIGHT, IMG_MAX_WIDTH), dtype=np.uint8)

        h, w = crop.shape
        scale = IMG_HEIGHT / max(h, 1)
        new_w = min(int(w * scale), IMG_MAX_WIDTH)
        new_w = max(new_w, 1)
        resized = cv2.resize(crop, (new_w, IMG_HEIGHT))

        canvas = np.full((IMG_HEIGHT, IMG_MAX_WIDTH), 255, dtype=np.uint8)
        canvas[:, :new_w] = resized

        tensor_img = torch.from_numpy(canvas).float().unsqueeze(0) / 255.0
        label = torch.tensor(encode_text(text), dtype=torch.long)

        return tensor_img, label


# ---------------- 模型（新增Dropout）----------------

class Encoder(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.5),          # 新增：CNN部分轻度dropout
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Dropout2d(dropout * 0.5),          # 新增
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(),
        )
        self.rnn = nn.LSTM(512, hidden_dim, bidirectional=True, batch_first=True, dropout=0)
        self.dropout = nn.Dropout(dropout)         # 新增：RNN输出后dropout

    def forward(self, x):
        conv_out = self.cnn(x)
        conv_out = conv_out.squeeze(2).permute(0, 2, 1)
        enc_out, (h, c) = self.rnn(conv_out)
        enc_out = self.dropout(enc_out)
        return enc_out


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        hidden_rep = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden_rep, encoder_outputs), dim=2)))
        attention_scores = self.v(energy).squeeze(2)
        return F.softmax(attention_scores, dim=1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)     # 新增
        self.attention = Attention(hidden_dim)
        self.rnn = nn.GRU(embed_dim + hidden_dim * 2, hidden_dim, batch_first=True)
        self.fc_dropout = nn.Dropout(dropout)          # 新增
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token, hidden, encoder_outputs):
        embedded = self.embedding(input_token).unsqueeze(1)
        embedded = self.embed_dropout(embedded)
        attn_weights = self.attention(hidden.squeeze(0), encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = self.fc_dropout(output)
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden


class AttentionOCR(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.encoder = Encoder(hidden_dim, dropout)
        self.decoder = Decoder(vocab_size, embed_dim, hidden_dim, dropout)
        self.hidden_dim = hidden_dim

    def forward(self, images, labels, teacher_forcing_ratio=0.5):
        batch_size = images.size(0)
        target_len = labels.size(1)
        vocab_size = self.decoder.fc.out_features

        encoder_outputs = self.encoder(images)
        hidden = torch.zeros(1, batch_size, self.hidden_dim, device=images.device)

        outputs = torch.zeros(batch_size, target_len, vocab_size, device=images.device)
        input_token = torch.full((batch_size,), SOS, dtype=torch.long, device=images.device)

        for t in range(1, target_len):
            output, hidden = self.decoder(input_token, hidden, encoder_outputs)
            outputs[:, t, :] = output
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            input_token = labels[:, t] if use_teacher_forcing else output.argmax(1)

        return outputs


# ---------------- 验证函数（新增） ----------------

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, labels, teacher_forcing_ratio=0.0)  # 验证时不用teacher forcing，更真实反映实际表现
            loss = criterion(
                outputs[:, 1:, :].reshape(-1, VOCAB_SIZE),
                labels[:, 1:].reshape(-1)
            )
            total_loss += loss.item()
    return total_loss / len(dataloader)


# ---------------- 训练主流程（新增验证+scheduler+early stopping） ----------------

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = SROIELineDataset(IMG_DIR, BOX_DIR)
    if len(full_dataset) == 0:
        raise RuntimeError("No samples loaded. Check IMG_DIR/BOX_DIR paths.")

    # 新增：拆分训练集/验证集
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Train samples: {train_size}, Validation samples: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = AttentionOCR(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 新增：学习率调度器，验证loss连续2轮不下降就自动降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    training_start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        epoch_start_time = time.time()

        # 新增：进度条，实时显示当前epoch跑到第几个batch、当前loss、速度
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, labels)

            loss = criterion(
                outputs[:, 1:, :].reshape(-1, VOCAB_SIZE),
                labels[:, 1:].reshape(-1)
            )
            loss.backward()

            # 新增：梯度裁剪，防止梯度爆炸导致训练不稳定
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            total_train_loss += loss.item()

            # 新增：进度条右侧实时显示当前batch的loss
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start_time
        total_elapsed = time.time() - training_start_time

        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - "
              f"Val Loss: {avg_val_loss:.4f} - LR: {current_lr:.6f} - "
              f"Epoch time: {epoch_time/60:.1f}min - Total elapsed: {total_elapsed/60:.1f}min")

        scheduler.step(avg_val_loss)

        # 新增：只在验证集表现变好时才保存模型（防止保存过拟合的版本）
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> Validation loss improved. Model saved.")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")

        # 新增：Early stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
            break

    print(f"Training done. Best model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
