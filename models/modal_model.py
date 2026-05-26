"""
模态参数预测模型。
DeepONet-like: Trunk(SIREN) → φ_k(x), Branch(MLP) → ω_k,ζ_k
PhysicsDecoder → 物理公式重建 FRF
"""
import torch
import torch.nn as nn
from .geometry_data import GeometryData
from .siren import SirenMLP


class PhysicsDecoder(nn.Module):
    """无参数物理解码器: φ + ω + ζ + φ_exc + freqs → FRF(Re,Im)

    H(x, x_f, ω) = Σ_k φ_k(x)·φ_k(x_f) / (ω_k²-ω²+j·2ζ_k·ω_k·ω)
    """

    def __init__(self, amp_scale: float = 500000.0,
                 freq_min: float = 1.0, freq_max: float = 8000.0):
        super().__init__()
        self.amp_scale = amp_scale
        self.freq_min = freq_min
        self.freq_max = freq_max

    def forward(self, phi, omega, zeta, frequencies, phi_exc=None, batch_idx=None):
        """
        phi:        (total_N, K) 可变N 或 (B, N, K) 同节点数
        omega:      (B, K)     rad/s 固有圆频率
        zeta:       (B, K)     阻尼比
        frequencies:(B, F)     归一化查询频率 [-1,1]
        phi_exc:    (B, K)     激励点振型值 φ_k(x_f)
        batch_idx:  (total_N,) 可变N时的批次索引
        """
        K = omega.shape[1]
        B, F = frequencies.shape
        var_n = (batch_idx is not None)

        f_phys = (frequencies + 1) / 2 * (self.freq_max - self.freq_min) + self.freq_min
        omega_q = 2.0 * torch.pi * f_phys  # (B, F)

        if var_n:
            total_N = phi.shape[0]
            frf_re = torch.zeros(total_N, F, device=phi.device)
            frf_im = torch.zeros(total_N, F, device=phi.device)
            for k in range(K):
                wk = omega[:, k]       # (B,)
                zk = zeta[:, k]        # (B,)
                pk = phi[:, k]         # (total_N,)
                if phi_exc is not None:
                    pk = pk * phi_exc[:, k][batch_idx]

                dw = wk.unsqueeze(1)**2 - omega_q**2  # (B, F)
                gamma = 2.0 * zk.unsqueeze(1) * wk.unsqueeze(1) * omega_q
                D = torch.clamp(dw**2 + gamma**2 + 1e-6, min=1.0)
                H_re = self.amp_scale * dw / D   # (B, F)
                H_im = -self.amp_scale * gamma / D

                frf_re += pk.unsqueeze(-1) * H_re[batch_idx]  # (total_N,1)*(total_N,F)
                frf_im += pk.unsqueeze(-1) * H_im[batch_idx]
            return torch.stack([frf_re, frf_im], dim=-1)  # (total_N, F, 2)

        else:
            N = phi.shape[1]
            frf_re = torch.zeros(B, N, F, device=phi.device)
            frf_im = torch.zeros(B, N, F, device=phi.device)
            for k in range(K):
                wk = omega[:, k]       # (B,)
                zk = zeta[:, k]        # (B,)
                pk = phi[:, :, k]      # (B, N)
                if phi_exc is not None:
                    pk = pk * phi_exc[:, k].unsqueeze(1)

                dw = wk.unsqueeze(1)**2 - omega_q**2  # (B, F)
                gamma = 2.0 * zk.unsqueeze(1) * wk.unsqueeze(1) * omega_q
                D = torch.clamp(dw**2 + gamma**2 + 1e-6, min=1.0)
                H_re = self.amp_scale * dw / D   # (B, F)
                H_im = -self.amp_scale * gamma / D

                frf_re += pk.unsqueeze(-1) * H_re.unsqueeze(1)  # (B,N,1)*(B,1,F)
                frf_im += pk.unsqueeze(-1) * H_im.unsqueeze(1)
            return torch.stack([frf_re, frf_im], dim=-1)  # (B, N, F, 2)


class ModalFRFModel(nn.Module):
    """
    Trunk(SIREN): (x,y,z) → spatial features → φ_k(x) 模态振型
    Branch(MLP):  point_features → global pool → ω_k, ζ_k 模态频率+阻尼
    PhysicsDecoder: φ + ω + ζ + freq → FRF(Re,Im)
    """

    def __init__(self, coord_dim=3, point_feat_dim=6, hidden_dim=256,
                 n_modes=2, trunk_layers=4, branch_layers=3, siren_w0=30.0,
                 amp_scale=500000.0, freq_min=1.0, freq_max=4000.0):
        super().__init__()
        self.n_modes = n_modes
        self.hidden_dim = hidden_dim

        # Trunk: 空间坐标 → SIREN编码
        self.trunk = SirenMLP(in_dim=coord_dim, hidden_dim=hidden_dim,
                              out_dim=hidden_dim, n_layers=trunk_layers, w0=siren_w0)

        # Branch: 逐点特征 → 全局池化
        branch_in = point_feat_dim if point_feat_dim > 0 else hidden_dim
        if point_feat_dim == 0:
            self.coord_proj = nn.Linear(coord_dim, hidden_dim)
        else:
            self.coord_proj = None

        branch_layers_list = []
        current_dim = branch_in if point_feat_dim > 0 else hidden_dim
        for i in range(branch_layers):
            out_d = hidden_dim if i < branch_layers - 1 else hidden_dim
            branch_layers_list.append(nn.Linear(current_dim, out_d))
            if i < branch_layers - 1:
                branch_layers_list.append(nn.LeakyReLU(0.1))
            current_dim = out_d
        self.branch = nn.Sequential(*branch_layers_list)

        # 输出头
        self.head_phi = nn.Linear(hidden_dim, n_modes)      # Trunk→模态振型
        self.head_modal = nn.Linear(hidden_dim, n_modes * 2)  # Branch→ω,ζ

        # 物理解码器
        self.physics = PhysicsDecoder(amp_scale=amp_scale,
                                       freq_min=freq_min, freq_max=freq_max)

    def _pool_to_global(self, x, batch_idx):
        """全局池化 → (B, hidden_dim)"""
        if batch_idx is not None:
            from torch_geometric.nn import global_mean_pool
            return global_mean_pool(x, batch_idx)  # (B, hidden_dim)
        return x.mean(dim=0, keepdim=True)  # (1, hidden_dim)

    def forward(self, geometry_data, frequencies=None, phi_exc=None):
        """
        训练forward: 预测模态参数 + 重建FRF
        返回: (frf_pred, omega_pred, zeta_pred, phi_pred)
        phi_exc: (B, K) 激励点振型值, 用于 FRF 重建
        """
        points = geometry_data.points
        point_feat = geometry_data.point_features
        batch_idx = geometry_data.batch
        B = points.shape[0] if points.ndim == 3 else 1

        # Trunk: 空间→隐特征
        var_n = points.ndim == 2  # (total_N, 3) → 可变节点数
        if var_n:
            spat = self.trunk(points)  # (total_N, hidden_dim)
        else:
            B, N, _ = points.shape
            pts_flat = points.reshape(-1, 3)
            spat = self.trunk(pts_flat).reshape(B, N, self.hidden_dim)  # (B, N, hidden_dim)

        # Branch: 全局特征→ω,ζ
        if point_feat is not None:
            br_in = point_feat
        elif self.coord_proj is not None:
            br_in = self.coord_proj(pts_flat).reshape(B, N, -1)
        else:
            br_in = torch.ones(B, N, self.hidden_dim, device=points.device)

        br_feat = self.branch(br_in)  # (B, N, H) or (total_N, H)
        if var_n:
            br_feat_flat = br_feat
        else:
            br_feat_flat = br_feat.reshape(-1, self.hidden_dim)
        macro = self._pool_to_global(br_feat_flat, batch_idx)  # (B, hidden_dim)

        # 输出头
        phi = self.head_phi(spat)  # (B, N, K)
        modal_out = self.head_modal(macro)  # (B, K*2)
        n_modes = self.n_modes
        omega = modal_out[:, :n_modes]  # (B, K)
        zeta = torch.nn.functional.softplus(modal_out[:, n_modes:]) * 0.001 + 1e-4

        # 物理重建FRF (仅评估时需要, 训练时frequencies=None跳过)
        if frequencies is not None:
            frf = self.physics(phi, omega, zeta, frequencies, phi_exc,
                               batch_idx=batch_idx if var_n else None)
        else:
            frf = None

        return frf, omega, zeta, phi
