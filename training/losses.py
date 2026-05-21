"""
losses.py — 频响函数 (FRF) 专用损失函数。

使用 Huber Loss (SmoothL1) 替代 MSE:
    - 误差小时为平方误差 (收敛平滑)
    - 误差大时退化为线性误差 (防梯度爆炸)
    特别适合 FRF 这种动态范围极大的数据。

同时支持共振峰自适应加权。
"""

import torch
import torch.nn.functional as F


def _huber(error, beta=1.0):
    """Huber 损失: |e|<=beta 用平方, |e|>beta 用线性"""
    abs_e = torch.abs(error)
    return torch.where(abs_e <= beta,
                       0.5 * error ** 2,
                       beta * (abs_e - 0.5 * beta))


def weighted_huber_loss(prediction, target, alpha=5.0, beta=1.0):
    """
    共振峰自适应加权 Huber Loss。

    weight = 1.0 + alpha * |target|  — 共振峰权重大
    内部用 Huber: 小误差平方收敛, 大误差线性防爆炸
    """
    weight = 1.0 + alpha * torch.abs(target)
    huber = _huber(prediction - target, beta)
    return torch.mean(weight * huber)


def complex_frf_loss(prediction, target, amp_weight=2.0, alpha=0.0, beta=1.0, eps=1e-8):
    """
    复数 FRF 损失 —— Huber 版。

    out_dim=2: loss = huber(Re) + huber(Im) + amp_weight * huber(|pred|, |target|)
    """
    pred_re, pred_im = prediction[..., 0], prediction[..., 1]
    target_re, target_im = target[..., 0], target[..., 1]

    pred_amp = torch.sqrt(pred_re ** 2 + pred_im ** 2 + eps)
    target_amp = torch.sqrt(target_re ** 2 + target_im ** 2 + eps)

    if alpha > 0:
        weight = 1.0 + alpha * target_amp
        loss_re = torch.mean(weight * _huber(pred_re - target_re, beta))
        loss_im = torch.mean(weight * _huber(pred_im - target_im, beta))
        loss_amp = torch.mean(weight * _huber(pred_amp - target_amp, beta))
    else:
        loss_re = F.smooth_l1_loss(pred_re, target_re, beta=beta)
        loss_im = F.smooth_l1_loss(pred_im, target_im, beta=beta)
        loss_amp = F.smooth_l1_loss(pred_amp, target_amp, beta=beta)

    return loss_re + loss_im + amp_weight * loss_amp


def frf_loss(prediction, target, out_dim=1, resonance_weight=2.0,
             amp_weight=2.0, freq_balance_strength=0.8):
    """
    FRF 统一损失函数 (MSE + 共振峰加权 + 频率平衡)。

    freq_balance_strength: 0=不平衡(第一峰主导), 0.8=强平衡(默认),
                          1.0=完全平等(每频率同等重要)
    """
    if out_dim == 2:
        return complex_frf_loss_mse(prediction, target,
                                    amp_weight=amp_weight, alpha=resonance_weight)
    else:
        weight = 1.0
        if resonance_weight > 0:
            weight = weight + resonance_weight * torch.abs(target)
        if freq_balance_strength > 0:
            # 频率平衡: 按目标幅值反比, 幅值小的频率获得更大权重
            freq_mean = torch.mean(torch.abs(target), dim=(0, 1))  # (F,)
            freq_w = 1.0 / (freq_mean + 1e-8)
            freq_w = freq_w / freq_w.mean()  # 归一化
            # 插值: 0=不用平衡, 1=完全平衡
            freq_w = 1.0 + freq_balance_strength * (freq_w - 1.0)
            weight = weight * freq_w
        return torch.mean(weight * (prediction - target) ** 2)


def complex_frf_loss_mse(prediction, target, amp_weight=2.0, alpha=0.0, eps=1e-8):
    """复数 FRF 损失 (MSE 版)"""
    pred_re, pred_im = prediction[..., 0], prediction[..., 1]
    target_re, target_im = target[..., 0], target[..., 1]
    pred_amp = torch.sqrt(pred_re ** 2 + pred_im ** 2 + eps)
    target_amp = torch.sqrt(target_re ** 2 + target_im ** 2 + eps)

    if alpha > 0:
        weight = 1.0 + alpha * target_amp
        loss_re = torch.mean(weight * (pred_re - target_re) ** 2)
        loss_im = torch.mean(weight * (pred_im - target_im) ** 2)
        loss_amp = torch.mean(weight * (pred_amp - target_amp) ** 2)
    else:
        loss_re = F.mse_loss(pred_re, target_re)
        loss_im = F.mse_loss(pred_im, target_im)
        loss_amp = F.mse_loss(pred_amp, target_amp)

    return loss_re + loss_im + amp_weight * loss_amp
