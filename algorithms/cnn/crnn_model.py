"""
经典 CRNN 结构 (Shi et al. 2015): CNN(卷积提取特征,高度逐步降到1) + BiLSTM(建模序列上下文) + FC输出。
输入: (B, 1, H=32, W)  单通道灰度图，固定高度32
输出: (W', B, num_classes)  给 CTCLoss 用的 log_softmax 序列
"""
import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes, img_height=32, hidden_size=256, in_channels=1):
        super().__init__()
        assert img_height % 16 == 0, "CRNN要求输入高度是16的倍数"

        # 卷积部分: 参考原论文结构, 高度32 -> 1, 宽度大致保留(下采样更少)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                   # 32x W -> 16 x W/2

            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                   # 16 x W/2 -> 8 x W/4

            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),                         # 8 x W/4 -> 4 x W/4 (只压高度)

            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),                         # 4 x W/4 -> 2 x W/4

            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            # 2 x W/4 -> 1 x (W/4 - 1)
        )

        self.rnn = nn.LSTM(512, hidden_size, num_layers=2, bidirectional=True,
                            batch_first=False, dropout=0.3)
        self.fc = nn.Linear(hidden_size * 2, num_classes + 1)  # +1 是CTC blank

    def forward(self, x):
        feat = self.cnn(x)                    # (B, C, H=1, W')
        assert feat.size(2) == 1, f"高度应该降到1, 实际是{feat.size(2)}, 检查输入高度是否是32"
        feat = feat.squeeze(2)                 # (B, C, W')
        feat = feat.permute(2, 0, 1)           # (W', B, C)  RNN要求 seq_len放第一维

        rnn_out, _ = self.rnn(feat)            # (W', B, hidden*2)
        out = self.fc(rnn_out)                 # (W', B, num_classes+1)
        return out.log_softmax(2)              # CTCLoss要求log概率