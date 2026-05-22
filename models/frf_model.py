"""
frf_model.py — 几何→FRF 完整模型。

组合几何编码器和逐点FRF解码器:
    GeometryData → GeometricEncoder → (B, N, D)
                                    → PerPointFRFDecoder + 频率查询 → (B, N, n_freqs)
"""

import torch
import torch.nn as nn
from .geometry_data import GeometryData
from .geometric_encoder import GeometricEncoder
from .perpoint_decoder import PerPointFRFDecoder


class GeometricFRFModel(nn.Module):
    """
    几何→频响函数 完整模型。

    使用方式:
        model = GeometricFRFModel(encoder, decoder)
        frf = model(geometry_data, frequencies)  # (B, N, n_freqs)
    """

    def __init__(self, encoder: GeometricEncoder, decoder: PerPointFRFDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, geometry_data: GeometryData,
                frequencies: torch.Tensor = None) -> torch.Tensor:
        """前向传播: 几何编码 → 逐点FRF解码"""
        point_features = self.encoder(geometry_data)         # (B, N, D)
        per_point_frf = self.decoder(point_features, frequencies)  # (B, N, n_freqs)
        return per_point_frf


def build_geometric_model(encoder_type: str = 'pointnet',
                          encoder_kwargs: dict = None,
                          decoder_kwargs: dict = None) -> GeometricFRFModel:
    """
    快速构建几何FRF模型。

    参数:
        encoder_type: 'simple' | 'pointnet' | 'gnn' | 'deeponet'
        encoder_kwargs: 编码器参数字典
        decoder_kwargs: 解码器参数字典 (out_dim=1振幅, out_dim=2复数FRF)

    返回:
        GeometricFRFModel 实例

    示例:
        # 仅振幅预测
        model = build_geometric_model('deeponet',
                    encoder_kwargs={'hidden_dim': 256, 'out_dim': 256},
                    decoder_kwargs={'in_dim': 256, 'n_freqs': 300, 'out_dim': 1})

        # 复数FRF (实部+虚部)
        model = build_geometric_model('deeponet',
                    encoder_kwargs={'hidden_dim': 256, 'out_dim': 256},
                    decoder_kwargs={'in_dim': 256, 'n_freqs': 300, 'out_dim': 2})
    """
    from .geometric_encoder import (
        SimplePointEncoder, PointNetEncoder, GNNEncoder, DeepONetEncoder
    )

    encoder_kwargs = encoder_kwargs or {}
    decoder_kwargs = decoder_kwargs or {}

    # 确保编码器输出维度与解码器输入维度一致
    encoder_out_dim = encoder_kwargs.get('out_dim', 256)
    decoder_kwargs.setdefault('in_dim', encoder_out_dim)

    encoder_registry = {
        'simple': SimplePointEncoder,
        'pointnet': PointNetEncoder,
        'gnn': GNNEncoder,
        'deeponet': DeepONetEncoder,
    }
    if encoder_type not in encoder_registry:
        raise ValueError(
            f"未知编码器类型: {encoder_type}，可选 {list(encoder_registry.keys())}"
        )

    encoder = encoder_registry[encoder_type](**encoder_kwargs)
    decoder = PerPointFRFDecoder(**decoder_kwargs)
    return GeometricFRFModel(encoder, decoder)
