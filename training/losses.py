"""
losses.py — 模态参数损失。各项初始值均衡到同量级。
"""
import torch
import torch.nn.functional as F


def modal_loss(omega_pred, omega_target,
               zeta_pred, zeta_target,
               phi_pred, phi_target):
    # omega ~40k rad/s → /40k → O(1), ×100 提升权重
    omega_ref = 40000.0
    loss_omega = F.mse_loss(omega_pred / omega_ref, omega_target / omega_ref) * 100.0
    # zeta ~0.003 → MSE~4e-6 → ×1e5 → 0.4
    loss_zeta  = F.mse_loss(zeta_pred, zeta_target) * 1e5
    # 质量归一化 φ ~[-5,5] → MSE~25 → ×1 → 25
    loss_phi   = F.mse_loss(phi_pred, phi_target) * 1.0
    return loss_omega + loss_zeta + loss_phi
