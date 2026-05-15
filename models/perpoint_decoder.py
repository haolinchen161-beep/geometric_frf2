"""
perpoint_decoder.py — 逐点频响函数 (FRF) 解码器。

将逐点隐特征 (B, N, D) 结合查询频率，通过 FiLM 调制 + MLP 解码出
每个点的完整频响曲线。

输出形状: (B, N, n_freqs, out_dim)
    - out_dim=1: 仅预测振幅 (Amplitude)
    - out_dim=2: 预测复数FRF的实部和虚部 (Real, Imag)，或振幅和相位

核心机制:
    - FiLM 频率查询: 频率值作为条件调制逐点特征
    - 分块(chunked)计算: 将点分批处理控制显存
"""

import torch
import torch.nn as nn
from .film import Film


class PerPointFRFDecoder(nn.Module):
    """
    逐点频响函数解码器。

    输入:  point_features (B, N, D) + frequencies (B, n_freqs) [可选]
    输出:  per_point_frf (B, N, n_freqs, out_dim)

    分块策略: chunk_size 控制一次处理多少个空间点，以时间换显存。
    """

    def __init__(self, in_dim: int = 256, n_freqs: int = 300,
                 hidden_dim: int = 256, n_layers: int = 3,
                 chunk_size: int = 512, out_dim: int = 1):
        """
        参数:
            in_dim:     输入逐点特征维度 (对应编码器输出维度)
            n_freqs:    输出频率点数
            hidden_dim: FiLM 调制后 MLP 的隐藏层宽度
            n_layers:   FiLM 调制后 MLP 的层数
            chunk_size: 分块大小（每次处理的点数）
            out_dim:    每个频率的输出维度。
                        1 = 仅振幅 (默认)
                        2 = 复数FRF (实部+虚部) 或 (振幅+相位)
        """
        super().__init__()
        self.in_dim = in_dim
        self.n_freqs = n_freqs
        self.chunk_size = chunk_size
        self.out_dim = out_dim

        # 频率查询 FiLM 层: 频率值(1维) → 调制参数
        self.queryfilm = Film(conditional_dim=1, projection_dim=in_dim)

        # 解码 MLP: 调制后特征 → FRF 值 (振幅 或 实部+虚部)
        layers = []
        current_dim = in_dim
        for i in range(n_layers):
            if i == n_layers - 1:
                # 最后一层: 输出 out_dim 个值 per 频率
                # out_dim=1: 振幅; out_dim=2: (实部, 虚部)
                layers.append(nn.Linear(current_dim, out_dim))
            else:
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.ReLU())
                current_dim = hidden_dim
        self.decoder_mlp = nn.Sequential(*layers)

        # 预计算归一化频率查询值 [-1, 1]
        self.register_buffer(
            'query_frequencies',
            torch.linspace(-1, 1, n_freqs).float()
        )

    def redefine_n_freqs(self, n_freqs: int):
        """动态调整输出频率点数"""
        self.n_freqs = n_freqs
        self.register_buffer(
            'query_frequencies',
            torch.linspace(-1, 1, n_freqs).float()
        )

    def forward(self, point_features: torch.Tensor,
                frequencies: torch.Tensor = None) -> torch.Tensor:
        """
        参数:
            point_features: (B, N, D) 几何编码器输出的逐点特征
            frequencies:    (B, n_freqs) 查询频率值，范围 [-1, 1]。
                            若为 None，使用预定义的均匀频率网格。

        返回:
            per_point_frf: 若 out_dim=1: (B, N, n_freqs)
                           若 out_dim>1: (B, N, n_freqs, out_dim)
        """
        B, N, D = point_features.shape
        n_freqs = frequencies.shape[1] if frequencies is not None else self.n_freqs

        all_chunks = []
        for start in range(0, N, self.chunk_size):
            end = min(start + self.chunk_size, N)
            chunk_features = point_features[:, start:end, :]  # (B, chunk_n, D)
            chunk_n = end - start

            # 展开: (B, chunk_n, D) → (B*chunk_n*n_freqs, D)
            feat_expanded = chunk_features.repeat_interleave(n_freqs, dim=1)
            feat_flat = feat_expanded.reshape(-1, D)

            # 构建频率查询
            if frequencies is not None:
                freq_repeated = frequencies.unsqueeze(1).expand(-1, chunk_n, -1)
                freq_flat = freq_repeated.reshape(-1, 1)
            else:
                qf = self.query_frequencies[:n_freqs].to(point_features.device)
                freq_flat = qf.repeat(B * chunk_n).view(-1, 1)

            # FiLM 调制 + MLP 解码
            modulated = self.queryfilm(feat_flat, freq_flat)
            decoded = self.decoder_mlp(modulated)  # (B*chunk_n*n_freqs, out_dim)

            # 恢复形状
            if self.out_dim == 1:
                chunk_frf = decoded.view(B, chunk_n, n_freqs)  # (B, chunk_n, n_freqs)
            else:
                chunk_frf = decoded.view(B, chunk_n, n_freqs, self.out_dim)  # (B, chunk_n, n_freqs, out_dim)
            all_chunks.append(chunk_frf)

        return torch.cat(all_chunks, dim=1)


class PerPointFRFDecoderSimple(nn.Module):
    """
    简化版逐点解码器 (无分块)。
    适用于点数较少的情况 (< 200)，计算更快但显存消耗大。
    """

    def __init__(self, in_dim: int = 256, n_freqs: int = 300,
                 hidden_dim: int = 256, n_layers: int = 3,
                 out_dim: int = 1):
        super().__init__()
        self.in_dim = in_dim
        self.n_freqs = n_freqs
        self.out_dim = out_dim

        self.queryfilm = Film(conditional_dim=1, projection_dim=in_dim)

        layers = []
        current_dim = in_dim
        for i in range(n_layers):
            if i == n_layers - 1:
                layers.append(nn.Linear(current_dim, out_dim))
            else:
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.ReLU())
                current_dim = hidden_dim
        self.decoder_mlp = nn.Sequential(*layers)

        self.register_buffer(
            'query_frequencies',
            torch.linspace(-1, 1, n_freqs).float()
        )

    def forward(self, point_features, frequencies=None):
        B, N, D = point_features.shape
        n_freqs = frequencies.shape[1] if frequencies is not None else self.n_freqs

        feat_expanded = point_features.repeat_interleave(n_freqs, dim=1)
        feat_flat = feat_expanded.reshape(-1, D)

        if frequencies is not None:
            freq_repeated = frequencies.unsqueeze(1).expand(-1, N, -1)
            freq_flat = freq_repeated.reshape(-1, 1)
        else:
            qf = self.query_frequencies[:n_freqs].to(point_features.device)
            freq_flat = qf.repeat(B * N).view(-1, 1)

        modulated = self.queryfilm(feat_flat, freq_flat)
        decoded = self.decoder_mlp(modulated)
        if self.out_dim == 1:
            return decoded.view(B, N, n_freqs)
        return decoded.view(B, N, n_freqs, self.out_dim)
