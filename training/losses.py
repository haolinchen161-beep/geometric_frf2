"""
losses.py — 模态参数损失 (MSPE 相对误差)。
"""
import torch
import torch.nn.functional as F


def modal_loss(omega_pred, omega_target,
               zeta_pred, zeta_target,
               phi_pred, phi_target):
    # ω: 0.2%误差→4e-6×5e5=2.0
    loss_omega = torch.mean((omega_pred - omega_target)**2
                            / (omega_target**2 + 1e-8)) * 500000.0
    # ζ: 5%误差→0.0025×1000=2.5
    loss_zeta  = torch.mean((zeta_pred - zeta_target)**2
                            / (zeta_target**2 + 1e-8)) * 1000.0
    # φ: MSE ~ O(1)
    loss_phi   = F.mse_loss(phi_pred, phi_target) * 1.0
    return loss_omega + loss_zeta + loss_phi


def phi_loss(phi_pred, phi_target):
    """阶段1: 只训振型"""
    return F.mse_loss(phi_pred, phi_target) * 1.0


def branch_loss(omega_pred, omega_target, zeta_pred, zeta_target):
    """阶段2: 只训ω/ζ"""
    loss_omega = torch.mean((omega_pred - omega_target)**2
                            / (omega_target**2 + 1e-8)) * 500000.0
    loss_zeta  = torch.mean((zeta_pred - zeta_target)**2
                            / (zeta_target**2 + 1e-8)) * 1000.0
    return loss_omega + loss_zeta


def frf_loss(frf_pred, frf_target):
    """阶段2 物理耦合: FRF重建误差 (asinh空间)"""
    return F.mse_loss(torch.asinh(frf_pred.clamp(-1e4, 1e4)), frf_target)
