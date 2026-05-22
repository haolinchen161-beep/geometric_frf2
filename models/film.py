"""
film.py — FiLM (Feature-wise Linear Modulation) 条件层。

FiLM 根据条件向量生成逐通道的缩放(weight)和平移(bias)参数，
用于将条件信息注入特征表示。

在逐点FRF解码器中，频率值作为条件，调制逐点隐特征以解码该频率下的响应。
"""

import torch.nn as nn


class Film(nn.Module):
    """
    FiLM 条件调制层。

    输入:
        x: (B, C, ...) 待调制的特征
        conditional: (B, D) 条件向量

    输出:
        weight(conditional) * x + bias(conditional)
        其中 weight, bias 均为 (B, C, 1, ...) 形状
    """

    def __init__(self, conditional_dim, projection_dim, **kwargs):
        super().__init__()
        self.weight = nn.Sequential(
            nn.Linear(conditional_dim, projection_dim, bias=False),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.bias = nn.Sequential(
            nn.Linear(conditional_dim, projection_dim, bias=False),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, x, conditional):
        ndim = len(x.shape) - 2
        view_shape = (x.shape[:2]) + (1,) * ndim
        return (self.weight(conditional).view(*view_shape) * x +
                self.bias(conditional).view(*view_shape))
