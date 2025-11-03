"""
BiFPN (Bidirectional Feature Pyramid Network)
用于YOLOv8的双向特征金字塔网络实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['WeightedFeatureFusion', 'BiFPNLayer']


class WeightedFeatureFusion(nn.Module):
    """
    加权特征融合模块
    使用可学习的权重对多个输入特征进行加权融合
    """

    def __init__(self, num_inputs=2, eps=1e-4):
        """
        Args:
            num_inputs: 输入特征的数量
            eps: 防止除零的小常数
        """
        super().__init__()
        self.num_inputs = num_inputs
        self.eps = eps
        # 可学习的权重参数
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))

    def forward(self, *inputs):
        """
        前向传播
        Args:
            *inputs: 多个输入特征张量
        Returns:
            融合后的特征张量
        """
        assert len(inputs) == self.num_inputs, \
            f"Expected {self.num_inputs} inputs, got {len(inputs)}"

        # ⭐ 修复：使用F.relu而不是self.relu，避免in-place操作
        # 确保权重为正
        weights = F.relu(self.weights)
        # 归一化权重
        weights = weights / (weights.sum() + self.eps)

        # 加权融合
        fused = sum(w * x for w, x in zip(weights, inputs))
        return fused


class BiFPNLayer(nn.Module):
    """
    简化版BiFPN层
    实现双向特征传递和加权融合
    """

    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: 输入通道数（第一个输入）
            out_channels: 输出通道数
        """
        super().__init__()

        from .conv import Conv

        # 为两个输入分别创建通道对齐卷积
        self.align1 = None
        self.align2 = None

        if in_channels != out_channels:
            self.align1 = Conv(in_channels, out_channels, 1)

        # 特征融合权重
        self.fusion = WeightedFeatureFusion(num_inputs=2)

        # 输出卷积
        self.conv = Conv(out_channels, out_channels, 3)

        self.out_channels = out_channels

    def forward(self, x1, x2):
        """
        前向传播
        Args:
            x1: 第一个输入特征
            x2: 第二个输入特征
        Returns:
            融合后的特征
        """
        # 调整x2的空间尺寸以匹配x1
        if x2.shape[2:] != x1.shape[2:]:
            x2 = F.interpolate(
                x2,
                size=x1.shape[2:],
                mode='nearest'
            )

        # 通道对齐x1
        if self.align1 is not None:
            x1 = self.align1(x1)

        # 通道对齐x2（如果需要）
        if x2.shape[1] != self.out_channels:
            # 动态创建align2（如果还没有）
            if self.align2 is None:
                from .conv import Conv
                self.align2 = Conv(x2.shape[1], self.out_channels, 1).to(x2.device)
            x2 = self.align2(x2)

        # 加权融合
        fused = self.fusion(x1, x2)

        # 输出卷积
        out = self.conv(fused)

        return out


# 用于测试
if __name__ == "__main__":
    print("Testing BiFPN modules...")

    # 测试WeightedFeatureFusion
    print("\n[测试1] WeightedFeatureFusion")
    fusion = WeightedFeatureFusion(num_inputs=2)
    x1 = torch.randn(1, 64, 80, 80)
    x2 = torch.randn(1, 64, 80, 80)
    out = fusion(x1, x2)
    print(f"✅ {x1.shape} + {x2.shape} -> {out.shape}")

    # 测试梯度
    x1.requires_grad = True
    x2.requires_grad = True
    out = fusion(x1, x2)
    loss = out.sum()
    loss.backward()
    print(f"✅ 梯度测试通过")

    # 测试BiFPNLayer
    print("\n[测试2] BiFPNLayer")
    bifpn = BiFPNLayer(in_channels=64, out_channels=128)
    x1 = torch.randn(1, 64, 80, 80, requires_grad=True)
    x2 = torch.randn(1, 64, 40, 40, requires_grad=True)
    out = bifpn(x1, x2)
    print(f"✅ {x1.shape} + {x2.shape} -> {out.shape}")

    # 测试梯度
    loss = out.sum()
    loss.backward()
    print(f"✅ 梯度测试通过")

    print("\n✅ 所有测试通过!")