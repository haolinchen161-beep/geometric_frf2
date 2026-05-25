"""
losses.py — 模态参数损失。纯 MSE, 量级自动平衡。
"""
import torch
import torch.nn.functional as F


def modal_loss(omega_pred, omega_target,
               zeta_pred, zeta_target,
               phi_pred, phi_target):
    loss_omega = F.mse_loss(omega_pred / 1000.0, omega_target / 1000.0)
    loss_zeta  = F.mse_loss(zeta_pred, zeta_target) * 1e5
    loss_phi   = F.mse_loss(phi_pred, phi_target) * 10.0
    return loss_omega + loss_zeta + loss_phi
