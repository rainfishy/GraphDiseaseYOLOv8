"""
Quality Focal Loss (QFL) Implementation
用于目标检测的质量感知损失函数

论文: Generalized Focal Loss v2: Learning Reliable Localization Quality Estimation for Dense Object Detection
链接: https://arxiv.org/abs/2011.12885
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QualityFocalLoss(nn.Module):
    """
    Quality Focal Loss (QFL)

    核心思想:
    - 将分类分数与IoU质量分数联合建模
    - 对高质量样本（高IoU）给予更大权重
    - 减少对低质量样本的过度惩罚
    - 使用连续的标签（0-1之间）而不是one-hot

    适用场景:
    - 小目标检测
    - 粘连目标检测
    - 密集目标检测
    """

    def __init__(self, use_sigmoid=True, beta=2.0, reduction='mean'):
        """
        Args:
            use_sigmoid: 是否对预测使用sigmoid
            beta: 调制因子，控制难易样本的权重分配
            reduction: 损失归约方式 ('none', 'mean', 'sum')
        """
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.beta = beta
        self.reduction = reduction

    def forward(self, pred, target, weight=None, avg_factor=None):
        """
        计算Quality Focal Loss

        Args:
            pred: 预测的分类分数，shape=[N, num_classes] 或 [N]
            target: 质量标签（0-1之间，结合了GT和IoU），shape与pred相同
            weight: 样本权重，shape=[N] (可选)
            avg_factor: 平均因子，用于归一化 (可选)

        Returns:
            loss: QFL损失值
        """
        # 确保pred和target在同一设备上
        if pred.device != target.device:
            target = target.to(pred.device)

        # 使用sigmoid将pred转换到[0,1]
        if self.use_sigmoid:
            pred_sigmoid = pred.sigmoid()
        else:
            pred_sigmoid = pred

        # 确保target也在[0,1]范围内
        target = target.float().clamp(0, 1)

        # 计算缩放因子: |target - pred_sigmoid|^beta
        # 当pred接近target时，scale_factor接近0，loss权重小
        # 当pred远离target时，scale_factor大，loss权重大
        scale_factor = (target - pred_sigmoid).abs().pow(self.beta)

        # 使用BCE作为基础损失
        # 这里使用with_logits版本以获得更好的数值稳定性
        if self.use_sigmoid:
            # 创建与target相同形状的全零tensor
            zerolabel = pred.new_zeros(pred.shape)

            # 计算BCE loss
            # 正样本: -log(sigmoid(pred))
            # 负样本: -log(1 - sigmoid(pred))
            loss = F.binary_cross_entropy_with_logits(
                pred, zerolabel, reduction='none'
            )

            # 添加正样本的额外项
            # QFL = |y - σ(z)|^β * BCE(z, 0) + y * (-log σ(z))
            positive_loss = target * F.logsigmoid(pred)
            loss = loss * scale_factor - positive_loss
        else:
            # 如果不使用sigmoid，直接计算BCE
            loss = F.binary_cross_entropy(
                pred_sigmoid, target, reduction='none'
            ) * scale_factor

        # 应用样本权重
        if weight is not None:
            if weight.dim() != loss.dim():
                if weight.dim() == 1:
                    weight = weight.view(-1, 1)
            loss = loss * weight

        # 损失归约
        if avg_factor is None:
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
        else:
            # 使用avg_factor进行归一化
            if self.reduction == 'mean':
                loss = loss.sum() / avg_factor
            elif self.reduction == 'sum':
                loss = loss.sum()

        return loss


class DistributionFocalLoss(nn.Module):
    """
    Distribution Focal Loss (DFL)

    用于边界框回归，将连续的回归问题转换为离散的分类问题
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target, weight=None):
        """
        Args:
            pred: 预测的分布，shape=[N, num_bins]
            target: 目标值（连续），shape=[N]
            weight: 样本权重，shape=[N]
        """
        # 将target转换为离散分布
        target = target.clamp(0, pred.size(1) - 1 - 0.01)
        target_left = target.long()
        target_right = target_left + 1

        # 计算插值权重
        weight_left = target_right.float() - target
        weight_right = target - target_left.float()

        # 对pred使用softmax
        pred_softmax = F.softmax(pred, dim=-1)

        # 提取对应位置的概率
        pred_left = pred_softmax.gather(1, target_left.unsqueeze(1)).squeeze(1)
        pred_right = pred_softmax.gather(
            1,
            target_right.clamp(max=pred.size(1) - 1).unsqueeze(1)
        ).squeeze(1)

        # 计算DFL损失
        loss = -(weight_left * torch.log(pred_left + 1e-10) +
                 weight_right * torch.log(pred_right + 1e-10))

        # 应用权重
        if weight is not None:
            loss = loss * weight

        # 归约
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


__all__ = ['QualityFocalLoss', 'DistributionFocalLoss']

# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("Testing Quality Focal Loss...")
    print("=" * 70)

    # 测试QFL
    qfl = QualityFocalLoss(beta=2.0)

    # 模拟数据
    batch_size = 16
    num_classes = 4

    # 预测分数（logits）
    pred = torch.randn(batch_size, num_classes)

    # 目标（结合了GT和IoU，范围0-1）
    # 高质量样本的target接近1，低质量样本接近0
    target = torch.rand(batch_size, num_classes)

    # 计算loss
    loss = qfl(pred, target)
    print(f"\n✅ QFL Loss: {loss.item():.4f}")

    # 测试梯度
    pred.requires_grad = True
    loss = qfl(pred, target)
    loss.backward()
    print(f"✅ Gradient shape: {pred.grad.shape}")
    print(f"✅ Gradient mean: {pred.grad.mean().item():.6f}")

    # 测试DFL
    print("\n" + "=" * 70)
    print("Testing Distribution Focal Loss...")
    print("=" * 70)

    dfl = DistributionFocalLoss()

    # 预测分布
    num_bins = 16
    pred_dist = torch.randn(batch_size, num_bins)

    # 目标位置（连续值）
    target_pos = torch.rand(batch_size) * (num_bins - 1)

    # 计算loss
    loss_dfl = dfl(pred_dist, target_pos)
    print(f"\n✅ DFL Loss: {loss_dfl.item():.4f}")

    # 测试梯度
    pred_dist.requires_grad = True
    loss_dfl = dfl(pred_dist, target_pos)
    loss_dfl.backward()
    print(f"✅ Gradient shape: {pred_dist.grad.shape}")

    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)