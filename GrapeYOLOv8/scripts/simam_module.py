"""
SimAM注意力机制实现
无参数、轻量级、高效的注意力模块
"""

import torch
import torch.nn as nn


class SimAM(nn.Module):
    """
    Simple, Parameter-Free Attention Module (SimAM)

    论文: SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks
    特点:
    - 无额外参数
    - 计算高效
    - 适合小目标检测
    """

    def __init__(self, e_lambda=1e-4):
        """
        参数:
            e_lambda: 能量函数的正则化参数
        """
        super(SimAM, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    @staticmethod
    def __get_module_name():
        return "SimAM"

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入特征图 [B, C, H, W]

        返回:
            输出特征图 [B, C, H, W]
        """
        b, c, h, w = x.size()

        # 计算每个通道的空间维度统计量
        # n: 空间维度的元素数量
        n = w * h - 1

        # 计算均值和方差
        # x_minus_mu_square: (x - μ)^2
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)

        # 计算能量函数 E
        # E_inv = 4 * (σ^2 + λ) / ((x - μ)^2 + 2σ^2 + 2λ)
        y = x_minus_mu_square / (
                4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
        ) + 0.5

        # 应用Sigmoid激活
        return x * self.activation(y)


class SimAM_Optimized(nn.Module):
    """
    优化版SimAM - 添加可选的通道维度处理
    """

    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM_Optimized, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1

        # 计算空间注意力（SimAM原始方法）
        x_mean = x.mean(dim=[2, 3], keepdim=True)
        x_var = (x - x_mean).pow(2).sum(dim=[2, 3], keepdim=True) / n

        # 计算能量
        x_minus_mu_square = (x - x_mean).pow(2)
        y = x_minus_mu_square / (4 * (x_var + self.e_lambda)) + 0.5

        # 应用注意力
        return x * self.activation(y)


# 测试代码
if __name__ == "__main__":
    # 测试SimAM模块
    print("=" * 70)
    print("🧪 测试SimAM注意力机制")
    print("=" * 70)

    # 创建模块
    simam = SimAM()

    # 创建测试输入 [batch, channels, height, width]
    x = torch.randn(2, 64, 32, 32)

    print(f"输入形状: {x.shape}")

    # 前向传播
    y = simam(x)

    print(f"输出形状: {y.shape}")
    print(f"参数量: {sum(p.numel() for p in simam.parameters())} (应为0)")

    # 检查输出
    assert x.shape == y.shape, "输入输出形状不匹配"
    print("✅ SimAM测试通过!")

    print("=" * 70)