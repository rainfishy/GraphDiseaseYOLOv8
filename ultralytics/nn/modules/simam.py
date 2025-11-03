"""SimAM: Simple, Parameter-Free Attention Module for YOLOv8"""

import torch
import torch.nn as nn

__all__ = ['SimAM']


class SimAM(nn.Module):
    """
    Simple, Parameter-Free Attention Module

    论文: SimAM: A Simple, Parameter-Free Attention Module for CNNs
    URL: https://proceedings.mlr.press/v139/yang21o.html

    特点:
        - 无需额外参数
        - 计算高效
        - 适合小目标检测
    """

    def __init__(self, e_lambda=1e-4):
        """
        Args:
            e_lambda (float): 能量函数的正则化参数
        """
        super(SimAM, self).__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        """
        前向传播

        Args:
            x (Tensor): 输入特征 [B, C, H, W]

        Returns:
            Tensor: 输出特征 [B, C, H, W]
        """
        b, c, h, w = x.size()
        n = w * h - 1

        # 计算 (x - μ)^2
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)

        # 计算能量函数
        y = x_minus_mu_square / (
                4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
        ) + 0.5

        # 应用注意力权重
        return x * self.act(y)
