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
    逐点频响函数解码器 (Concat模式)。

    将逐点特征与频率值拼接后通过共享MLP解码出每个点的FRF。
    相比原FiLM方案，直接拼接频率值到特征中，MLP可自由学习
    频率与空间特征的联合映射，表达能力更强。

    输入:  point_features (B, N, D) + frequencies (B, n_freqs) [可选]
    输出:  per_point_frf (B, N, n_freqs, out_dim)

    分块策略: chunk_size 控制一次处理多少个空间点，以时间换显存。
    """

    def __init__(self, in_dim: int = 256, n_freqs: int = 300,
                 hidden_dim: int = 256, n_layers: int = 3,
                 chunk_size: int = 128, out_dim: int = 1,
                 freq_encoding: str = 'none'):
        """
        参数:
            in_dim:       输入逐点特征维度
            n_freqs:      输出频率点数
            hidden_dim:   MLP 隐藏层宽度
            n_layers:     MLP 层数
            chunk_size:   分块大小
            out_dim:      每频率输出维度 (1=有符号位移FRF, 2=复数)
            freq_encoding: 频率编码方式:
                'none'  — 直接拼接标量频率值 [默认]
                'sin'   — sin + cos 编码 (帮助MLP学习高频周期特征)
        """
        super().__init__()
        self.in_dim = in_dim
        self.n_freqs = n_freqs
        self.chunk_size = chunk_size
        self.out_dim = out_dim
        self.freq_encoding = freq_encoding

        # 频率编码维度
        if freq_encoding == 'sin':
            self.freq_enc_dim = 64  # sin/cos at 32 scales, 让频率信号不被空间特征淹没
            freq_input_dim = in_dim + self.freq_enc_dim
        else:
            self.freq_enc_dim = 0
            freq_input_dim = in_dim + 1  # +1 = 频率标量

        # 解码 MLP: [point_feat, freq_enc] → FRF 值
        layers = []
        current_dim = freq_input_dim
        for i in range(n_layers):
            if i == n_layers - 1:
                layers.append(nn.Linear(current_dim, out_dim))
            else:
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.LeakyReLU(0.1))  # 替代ReLU: 负值不截断, 高频信号可传播
                current_dim = hidden_dim
        self.decoder_mlp = nn.Sequential(*layers)

        # 预计算归一化频率值 [-1, 1]
        self.register_buffer(
            'query_frequencies',
            torch.linspace(-1, 1, n_freqs).float()
        )

    def _encode_frequency(self, freq_flat: torch.Tensor) -> torch.Tensor:
        """频率编码: 标量 → 高维特征 (32 个 sin/cos 尺度)"""
        if self.freq_encoding == 'sin':
            # 指数增长的尺度覆盖低频到高频
            scales = 2.0 ** torch.arange(0, 5, 0.15625, device=freq_flat.device)  # 32 个尺度
            freq_scaled = freq_flat * scales.unsqueeze(0)  # (N, 32)
            return torch.cat([
                torch.sin(freq_scaled * torch.pi),
                torch.cos(freq_scaled * torch.pi),
            ], dim=-1)  # (N, 64)
        else:
            return freq_flat  # (N, 1)

    def redefine_n_freqs(self, n_freqs: int):
        self.n_freqs = n_freqs
        self.register_buffer(
            'query_frequencies',
            torch.linspace(-1, 1, n_freqs).float()
        )

    def forward(self, point_features: torch.Tensor,
                frequencies: torch.Tensor = None) -> torch.Tensor:
        """
        参数:
            point_features: (B, N, D)
            frequencies:    (B, n_freqs), 范围 [-1, 1]

        返回:
            若 out_dim=1: (B, N, n_freqs)
            若 out_dim>1: (B, N, n_freqs, out_dim)
        """
        B, N, D = point_features.shape
        n_freqs = frequencies.shape[1] if frequencies is not None else self.n_freqs

        all_chunks = []
        for start in range(0, N, self.chunk_size):
            end = min(start + self.chunk_size, N)
            chunk_features = point_features[:, start:end, :]  # (B, chunk_n, D)
            chunk_n = end - start

            # 展开为 (B, chunk_n, n_freqs, D) 并拼接频率
            # point: (B, chunk_n, 1, D) → (B, chunk_n, n_freqs, D)
            pf = chunk_features.unsqueeze(2).expand(-1, -1, n_freqs, -1)

            # freq: (B, 1, n_freqs, 1) → (B, chunk_n, n_freqs, 1)
            if frequencies is not None:
                freq = frequencies
            else:
                freq = self.query_frequencies[:n_freqs].to(point_features.device)
                freq = freq.unsqueeze(0).expand(B, -1)
            ff_raw = freq.unsqueeze(1).expand(-1, chunk_n, -1)  # (B, chunk_n, n_freqs)

            # 频率编码
            ff_flat = ff_raw.reshape(-1, 1)  # (B*chunk_n*n_freqs, 1)
            ff_enc = self._encode_frequency(ff_flat)  # (B*chunk_n*n_freqs, enc_dim)

            # 拼接 [point_features, freq_encoding] → (B*chunk_n*n_freqs, D+enc_dim)
            pf_flat = pf.reshape(-1, D)
            combined = torch.cat([pf_flat, ff_enc], dim=-1)

            # MLP 解码
            decoded = self.decoder_mlp(combined)  # (B*chunk_n*n_freqs, out_dim)

            # 恢复形状
            if self.out_dim == 1:
                chunk_frf = decoded.view(B, chunk_n, n_freqs)
            else:
                chunk_frf = decoded.view(B, chunk_n, n_freqs, self.out_dim)
            all_chunks.append(chunk_frf)

        return torch.cat(all_chunks, dim=1)


class PerPointFRFDecoderSimple(nn.Module):
    """
    简化版逐点解码器 (无分块, Concat模式)。
    适用于点数较少的情况 (< 200)，计算更快但显存消耗大。
    """

    def __init__(self, in_dim: int = 256, n_freqs: int = 300,
                 hidden_dim: int = 256, n_layers: int = 3,
                 out_dim: int = 1):
        super().__init__()
        self.in_dim = in_dim
        self.n_freqs = n_freqs
        self.out_dim = out_dim

        # MLP: [point_feat + freq] → FRF
        layers = []
        current_dim = in_dim + 1  # +1 for frequency scalar
        for i in range(n_layers):
            if i == n_layers - 1:
                layers.append(nn.Linear(current_dim, out_dim))
            else:
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.LeakyReLU(0.1))  # 替代ReLU: 负值不截断, 高频信号可传播
                current_dim = hidden_dim
        self.decoder_mlp = nn.Sequential(*layers)

        self.register_buffer(
            'query_frequencies',
            torch.linspace(-1, 1, n_freqs).float()
        )

    def forward(self, point_features, frequencies=None):
        B, N, D = point_features.shape
        n_freqs = frequencies.shape[1] if frequencies is not None else self.n_freqs

        # 拼接: (B, N, 1, D) + (B, 1, n_freqs, 1) → (B, N, n_freqs, D+1)
        pf = point_features.unsqueeze(2).expand(-1, -1, n_freqs, -1)
        if frequencies is not None:
            freq = frequencies
        else:
            freq = self.query_frequencies[:n_freqs].to(point_features.device)
            freq = freq.unsqueeze(0).expand(B, -1)
        ff = freq.unsqueeze(1).unsqueeze(-1).expand(-1, N, -1, 1)
        combined = torch.cat([pf, ff], dim=-1)  # (B, N, n_freqs, D+1)

        decoded = self.decoder_mlp(combined)
        if self.out_dim == 1:
            return decoded.squeeze(-1)  # (B, N, n_freqs)
        return decoded  # (B, N, n_freqs, out_dim)
