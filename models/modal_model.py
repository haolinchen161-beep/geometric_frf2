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
                 amp_scale=500000.0, freq_min=1.0, freq_max=5000.0,
                 branch_hidden_dim=64):
        super().__init__()
        self.n_modes = n_modes
        self.hidden_dim = hidden_dim
        bh = branch_hidden_dim
        self.bh = bh

        # Trunk: 空间坐标 → SIREN编码
        self.trunk = SirenMLP(in_dim=coord_dim, hidden_dim=hidden_dim,
                              out_dim=hidden_dim, n_layers=trunk_layers, w0=siren_w0)

        # Branch: 逐点特征 → 全局池化 (7维含物理先验, 轻量)
        branch_in = point_feat_dim + 1  # +1 = physics prior
        if point_feat_dim == 0:
            self.coord_proj = nn.Linear(coord_dim, bh)
        else:
            self.coord_proj = None
        if point_feat_dim == 0:
            self.coord_proj = nn.Linear(coord_dim, hidden_dim)
        else:
            self.coord_proj = None

        branch_layers_list = []
        current_dim = branch_in
        for i in range(branch_layers):
            out_d = bh if i < branch_layers - 1 else bh
            branch_layers_list.append(nn.Linear(current_dim, out_d))
            if i < branch_layers - 1:
                branch_layers_list.append(nn.LeakyReLU(0.1))
            current_dim = out_d
        self.branch = nn.Sequential(*branch_layers_list)

        # 跳跃连接: 物理先验直达ω, MLP只学修正
        self.skip_omega = nn.Linear(1, n_modes)               # (H/L²)√(E/ρ)→ω基线
        nn.init.constant_(self.skip_omega.weight, 0.0)
        nn.init.constant_(self.skip_omega.bias, 0.0)          # init: ω_coarse≈13860≈中位ω

        # 输出头
        self.head_phi = nn.Linear(hidden_dim, n_modes)      # Trunk→模态振型
        self.head_modal = nn.Linear(bh, n_modes * 2)         # Branch修正→Δω,ζ

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

        # 坐标归一化: 绝对坐标→[-1,1], SIREN只学形状, Branch管尺度
        var_n = points.ndim == 2  # (total_N, 3) → 可变节点数
        if var_n:
            # 每个样本独立归一化
            pts_norm = []
            for b in range(batch_idx.max().item() + 1):
                mask = batch_idx == b
                p_b = points[mask]  # (N_b, 3)
                lo, hi = p_b.min(dim=0, keepdim=True)[0], p_b.max(dim=0, keepdim=True)[0]
                pts_norm.append((p_b - lo) / (hi - lo + 1e-8) * 2.0 - 1.0)
            pts_siren = torch.cat(pts_norm, dim=0)  # (total_N, 3)
        else:
            B, N, _ = points.shape
            pts_norm = []
            for b in range(B):
                p_b = points[b]  # (N, 3)
                lo, hi = p_b.min(dim=0, keepdim=True)[0], p_b.max(dim=0, keepdim=True)[0]
                pts_norm.append((p_b - lo) / (hi - lo + 1e-8) * 2.0 - 1.0)
            pts_siren = torch.stack(pts_norm, dim=0)  # (B, N, 3)

        # Trunk: 归一化坐标→空间隐特征
        if var_n:
            spat = self.trunk(pts_siren)  # (total_N, hidden_dim)
        else:
            pts_flat = pts_siren.reshape(-1, 3)
            spat = self.trunk(pts_flat).reshape(B, N, self.hidden_dim)  # (B, N, hidden_dim)

        # Branch: 全局特征→ω,ζ (注入物理先验: ω∝(H/L²)√(E/ρ))
        if point_feat is not None:
            E_r, rho_r, L_r, _, H_r = [point_feat[..., i] for i in [0, 1, 2, 3, 4]]
            physics = (H_r / (L_r**2 + 1e-6)) * torch.sqrt(torch.abs(E_r / (rho_r + 1e-6)))
            br_in = torch.cat([point_feat, physics.unsqueeze(-1)], dim=-1)
        elif self.coord_proj is not None:
            br_in = self.coord_proj(pts_flat).reshape(B, N, -1)
        else:
            br_in = torch.ones(B, N, self.hidden_dim, device=points.device)

        br_feat = self.branch(br_in)  # (B, N, bh) or (total_N, bh)
        if var_n:
            br_feat_flat = br_feat
        else:
            br_feat_flat = br_feat.reshape(-1, self.bh)
        macro = self._pool_to_global(br_feat_flat, batch_idx)  # (B, hidden_dim)

        # 输出头
        phi = self.head_phi(spat)  # (B, N, K)
        modal_out = self.head_modal(macro)  # (B, K*2)
        n_modes = self.n_modes

        # 跳跃连接: 物理先验→ω基线, MLP→修正 (每样本取第一个节点的physics)
        if var_n:
            num_b = int(batch_idx.max().item()) + 1
            physics_sample = torch.zeros(num_b, device=points.device)
            for b in range(num_b):
                physics_sample[b] = physics[batch_idx == b][0]
        else:
            physics_sample = physics[:, 0] if physics.ndim == 2 else physics.unsqueeze(0)[:, 0]
        omega_base = self.skip_omega(physics_sample.unsqueeze(-1))[:, :n_modes]  # (B, K)
        # 粗-细解耦: skip承担宏观(×40000), MLP只做±2000微调 (消除梯度放大)
        omega_coarse = torch.nn.functional.softplus(omega_base) * 20000.0
        omega_fine = torch.tanh(modal_out[:, :n_modes]) * 8000.0
        omega = omega_coarse + omega_fine
        # softplus(0)≈0.693 → 0.693*0.004+1e-4≈0.00287≈target 0.003
        zeta = torch.nn.functional.softplus(modal_out[:, n_modes:]) * 0.004 + 1e-4

        # 物理重建FRF (仅评估时需要, 训练时frequencies=None跳过)
        if frequencies is not None:
            frf = self.physics(phi, omega, zeta, frequencies, phi_exc,
                               batch_idx=batch_idx if var_n else None)
        else:
            frf = None

        return frf, omega, zeta, phi
