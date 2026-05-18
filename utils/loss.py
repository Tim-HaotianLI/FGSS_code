import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha = None, gamma = 2, reduction="mean"):
        super().__init__();
        self.alpha = alpha  # 类别权重
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)  # 预测正确的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)

        # 计算smooth targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)

            if self.weight is not None:
                true_dist = true_dist * self.weight[target].unsqueeze(1)

        return torch.mean(torch.sum(-true_dist * log_preds, dim=-1))


class FocalLossWithMining(nn.Module):
    """带难样本挖掘的Focal Loss"""

    def __init__(self, alpha=None, gamma=2.0, mining_ratio=0.7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.mining_ratio = mining_ratio

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # 难样本挖掘：只关注loss最大的样本
        num_hard = int(focal_loss.size(0) * self.mining_ratio)
        hard_losses, _ = torch.topk(focal_loss, num_hard)

        return hard_losses.mean()

class CombinedLoss(nn.Module):
    def __init__(self, alpha = None, gamma=2.0, smoothing=0.1):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        # self.smooth_ce = LabelSmoothingCrossEntropy(smoothing=smoothing, weight=alpha)
        self.ce = nn.CrossEntropyLoss(weight=alpha)

    def forward(self, inputs, targets):
        return 0.7 * self.focal(inputs, targets) + 0.3 * self.ce(inputs, targets)


class OptimalCombinedLoss(nn.Module):
    """优化的组合损失"""

    def __init__(self, alpha=None, gamma=2.0, smoothing=0.15, mining_ratio=0.7):
        super().__init__()
        self.focal_mining = FocalLossWithMining(alpha, gamma, mining_ratio)
        self.label_smooth = LabelSmoothingCrossEntropy(smoothing, alpha)

        # 动态权重
        self.focal_weight = nn.Parameter(torch.tensor(0.7))
        self.smooth_weight = nn.Parameter(torch.tensor(0.3))

    def forward(self, inputs, targets):
        focal_loss = self.focal_mining(inputs, targets)
        smooth_loss = self.label_smooth(inputs, targets)

        # 使用可学习的权重
        focal_w = torch.sigmoid(self.focal_weight)
        smooth_w = torch.sigmoid(self.smooth_weight)

        total = focal_w * focal_loss + smooth_w * smooth_loss
        return total / (focal_w + smooth_w)  # 归一化


class SimpleLabelSmoothingLoss(nn.Module):
    """简化的Label Smoothing - 最适合小数据集"""

    def __init__(self, class_weights, smoothing=0.2):
        super().__init__()
        self.class_weights = class_weights
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)

        # Smooth targets
        with torch.no_grad():
            true_dist = torch.zeros_like(log_preds)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        # 应用class weights
        loss = -true_dist * log_preds
        loss = loss.sum(dim=-1)

        # 🔥 关键：按样本的真实类别加权
        if self.class_weights is not None:
            loss = loss * self.class_weights[target]

        return loss.mean()


class SimpleFocalLoss(nn.Module):
    """简化的Focal Loss - 去掉难样本挖掘"""

    def __init__(self, class_weights, gamma=1.5):  # 🔥 gamma降到1.5
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma

    def forward(self, inputs, targets):
        # 标准CE loss
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.class_weights,
            reduction='none'
        )

        # Focal term
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


class FixedCombinedLoss(nn.Module):
    """固定权重的组合损失"""

    def __init__(self, alpha, gamma=1.5, smoothing=0.1,
                 focal_weight=0.6, smooth_weight=0.4):
        super().__init__()
        self.focal = SimpleFocalLoss(alpha, gamma)
        self.smooth = SimpleLabelSmoothingLoss(alpha, smoothing)

        # 🔥 固定权重，不可学习
        self.focal_weight = focal_weight
        self.smooth_weight = smooth_weight

    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        smooth_loss = self.smooth(inputs, targets)

        return (self.focal_weight * focal_loss +
                self.smooth_weight * smooth_loss)