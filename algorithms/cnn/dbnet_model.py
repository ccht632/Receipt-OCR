"""
DBNet 模型结构: ResNet backbone(ImageNet预训练) + FPN neck + DB head
输出三张图: prob_map(文字概率图), thresh_map(阈值图), binary_map(可微二值化图)
推理时只需要 prob_map 去做后处理提取文字框。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResNetBackbone(nn.Module):
    """提取4个stage的特征图，stride分别是4/8/16/32。"""

    def __init__(self, backbone="resnet18", pretrained=True):
        super().__init__()
        if backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self.out_channels = [64, 128, 256, 512]
        elif backbone == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            self.out_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"不支持的backbone: {backbone}")

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1  # stride 4
        self.layer2 = net.layer2  # stride 8
        self.layer3 = net.layer3  # stride 16
        self.layer4 = net.layer4  # stride 32

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


class FPN(nn.Module):
    """标准FPN: 1x1降通道 -> 上采样相加 -> 3x3平滑 -> 全部上采样到1/4分辨率拼接。"""

    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.reduce = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels_list
        ])
        self.smooth = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels // 4, 3, padding=1) for _ in in_channels_list
        ])

    def forward(self, feats):
        c2, c3, c4, c5 = feats
        p5 = self.reduce[3](c5)
        p4 = self.reduce[2](c4) + F.interpolate(p5, size=c4.shape[2:], mode="nearest")
        p3 = self.reduce[1](c3) + F.interpolate(p4, size=c3.shape[2:], mode="nearest")
        p2 = self.reduce[0](c2) + F.interpolate(p3, size=c2.shape[2:], mode="nearest")

        target_size = p2.shape[2:]
        f5 = F.interpolate(self.smooth[3](p5), size=target_size, mode="nearest")
        f4 = F.interpolate(self.smooth[2](p4), size=target_size, mode="nearest")
        f3 = F.interpolate(self.smooth[1](p3), size=target_size, mode="nearest")
        f2 = self.smooth[0](p2)

        fused = torch.cat([f2, f3, f4, f5], dim=1)  # 通道数 = out_channels (4 * out_channels//4)
        return fused


class DBHead(nn.Module):
    """概率图分支 + 阈值图分支，各自两次上采样恢复到原图1/1分辨率(相对stride4的特征图再上采样4倍)。"""

    def __init__(self, in_channels=256, k=50):
        super().__init__()
        self.k = k
        self.prob_head = self._build_head(in_channels)
        self.thresh_head = self._build_head(in_channels)

    @staticmethod
    def _build_head(in_channels):
        mid = in_channels // 4
        return nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid, mid, 2, stride=2),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid, 1, 2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        prob_map = self.prob_head(x)
        thresh_map = self.thresh_head(x)
        binary_map = torch.reciprocal(1 + torch.exp(-self.k * (prob_map - thresh_map)))
        return prob_map, thresh_map, binary_map


class DBNet(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True, fpn_channels=256):
        super().__init__()
        self.backbone = ResNetBackbone(backbone, pretrained)
        self.fpn = FPN(self.backbone.out_channels, fpn_channels)
        self.head = DBHead(fpn_channels)

    def forward(self, x):
        feats = self.backbone(x)
        fused = self.fpn(feats)
        prob_map, thresh_map, binary_map = self.head(fused)
        return prob_map, thresh_map, binary_map