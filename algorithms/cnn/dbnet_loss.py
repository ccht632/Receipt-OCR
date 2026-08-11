"""
DBNet 损失函数，标准DB论文组合:
  L = Lprob(OHEM二分类BCE) + alpha * Lbinary(Dice) + beta * Lthresh(MaskL1)
OHEM: 正样本(文字)远少于负样本(背景)，只取loss最大的一部分负样本参与计算，避免负样本淹没梯度。
"""
import torch
import torch.nn as nn


class BalanceCrossEntropyLoss(nn.Module):
    def __init__(self, negative_ratio=3.0, eps=1e-6):
        super().__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self, pred, gt, mask):
        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()), int(positive_count * self.negative_ratio))
        negative_count = max(negative_count, 1)

        loss = nn.functional.binary_cross_entropy(pred, gt, reduction="none")
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()

        negative_loss_flat = negative_loss.view(-1)
        negative_loss_topk, _ = torch.topk(negative_loss_flat, negative_count)

        balance_loss = (positive_loss.sum() + negative_loss_topk.sum()) / \
                        (positive_count + negative_count + self.eps)
        return balance_loss


class MaskL1Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt, mask):
        loss = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        return loss


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt, mask):
        intersection = (pred * gt * mask).sum()
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        return 1 - 2 * intersection / union


class DBLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=10.0):
        super().__init__()
        self.alpha = alpha  # binary_map(Dice) 权重
        self.beta = beta    # thresh_map(L1) 权重
        self.prob_loss_fn = BalanceCrossEntropyLoss()
        self.thresh_loss_fn = MaskL1Loss()
        self.binary_loss_fn = DiceLoss()

    def forward(self, prob_map, thresh_map, binary_map, batch):
        prob_gt = batch["prob_gt"]
        prob_mask = batch["prob_mask"]
        thresh_gt = batch["thresh_gt"]
        thresh_mask = batch["thresh_mask"]

        loss_prob = self.prob_loss_fn(prob_map, prob_gt, prob_mask)
        loss_thresh = self.thresh_loss_fn(thresh_map, thresh_gt, thresh_mask)
        loss_binary = self.binary_loss_fn(binary_map, prob_gt, prob_mask)

        total = loss_prob + self.alpha * loss_binary + self.beta * loss_thresh
        return total, {
            "loss_prob": loss_prob.item(),
            "loss_thresh": loss_thresh.item(),
            "loss_binary": loss_binary.item(),
        }