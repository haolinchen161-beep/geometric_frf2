"""
trainer.py — 逐点 FRF 训练循环 + 评估。

数据流:
    geometry + frequencies → net → per_point_frf (B, N, n_freqs[, out_dim])
    损失: FRF 专用损失函数 (共振峰自适应加权 + 复数物理约束)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from .losses import frf_loss


def train(args, config, model_cfg, net, dataloader, optimizer,
          valloader, scheduler, logger=None, start_epoch=0):
    """
    逐点 FRF 训练循环。

    参数:
        start_epoch: 起始轮次 (0=从零训练, >0=续训)
    """
    lowest = np.inf
    net.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # 续训时把 scheduler 状态推进到当前轮次
    if start_epoch > 0 and scheduler is not None:
        scheduler.last_epoch = start_epoch
        # CosineAnnealingLR 闭式解: 直接计算 epoch=last_epoch 时的 LR
        import math
        for param_group in scheduler.optimizer.param_groups:
            base_lr = param_group['initial_lr']
            t = scheduler.last_epoch / scheduler.T_max
            param_group['lr'] = scheduler.eta_min + (base_lr - scheduler.eta_min) * 0.5 * (1.0 + math.cos(math.pi * t))

    for epoch in range(start_epoch, config.get('epochs', 500)):
        losses = []
        for batch in dataloader:
            optimizer.zero_grad()

            geometry = batch['geometry'].to(args.device)
            target = batch['point_frf'].to(args.device)
            frequencies = batch['frequencies'].to(args.device)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                prediction = net(geometry, frequencies)

                out_dim = model_cfg.get('decoder_kwargs', {}).get('out_dim', 1)
                resonance_weight = config.get('resonance_weight', 2.0)
                amp_weight = config.get('amp_weight', 2.0)
                freq_balance = config.get('freq_balance_strength', 0.8)
                loss = frf_loss(prediction, target,
                               out_dim=out_dim,
                               resonance_weight=resonance_weight,
                               amp_weight=amp_weight,
                               freq_balance_strength=freq_balance)

                # —— 边界物理约束: 固定端 (x≈0) 位移必须为零 ——
                # 检测固定端节点: x坐标小于梁总长的 2%
                pts = geometry.points
                x_max = pts[..., 0].max() if pts.ndim == 3 else pts[:, 0].max()
                is_fixed = (pts[..., 0] < x_max * 0.02).float()  # (B, N) 或 (total_N,)
                # 扩展掩码匹配 prediction 形状
                if out_dim == 1:
                    boundary_mask = is_fixed.unsqueeze(-1)  # (B, N, 1)
                else:
                    boundary_mask = is_fixed.unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
                n_fixed = boundary_mask.sum() + 1e-8
                boundary_loss = (boundary_mask * prediction ** 2).sum() / n_fixed
                loss = loss + 50.0 * boundary_loss

            losses.append(loss.detach().cpu().item())

            scaler.scale(loss).backward()

            grad_clip = config.get('optimizer', {}).get('gradient_clip')
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

        if scheduler is not None:
            scheduler.step()

        mean_loss = np.mean(losses)
        _log(f"Epoch {epoch} training loss = {mean_loss:4.4}", logger)
        _wandb_log({'Loss / Training': mean_loss, 'LR': optimizer.param_groups[0]['lr'], 'Epoch': epoch}, logger)

        val_freq = config.get('validation_frequency', 5)
        if epoch % val_freq == 0 or epoch % int(config.get('epochs', 500) / 10) == 0:
            save_model(args.dir, epoch, net, optimizer, loss, "checkpoint_last")
            val_loss = evaluate(args, config, net, valloader, logger, epoch)["loss (test/val)"]
            if val_loss < lowest:
                _log("best model", logger)
                save_model(args.dir, epoch, net, optimizer, val_loss)
                lowest = val_loss

        if epoch == (config.get('epochs', 500) - 1):
            path = os.path.join(args.dir, "checkpoint_best")
            net.load_state_dict(torch.load(path)["model_state_dict"])
            evaluate(args, config, net, valloader, logger, epoch, verbose=True)

    return net


def evaluate(args, config, net, dataloader, logger=None, epoch=None, verbose=True):
    """验证/测试评估"""
    prediction, output = _generate_preds(args, config, net, dataloader)
    results = _evaluate(prediction, output, logger, epoch, verbose)
    return results


def _generate_preds(args, config, net, dataloader):
    """生成所有预测"""
    net.eval()
    with torch.no_grad():
        predictions, outputs = [], []
        for batch in dataloader:
            geometry = batch['geometry'].to(args.device)
            target = batch['point_frf'].to(args.device)
            frequencies = batch['frequencies'].to(args.device)
            prediction = net(geometry, frequencies)

            max_freq = config.get('max_frequency')
            if max_freq is not None:
                prediction = prediction[:, :, :max_freq]
                target = target[:, :, :max_freq]

            predictions.append(prediction.detach().cpu())
            outputs.append(target.detach().cpu())

    return torch.cat(predictions, dim=0), torch.cat(outputs, dim=0)


def _evaluate(prediction, output, logger, epoch, verbose=True):
    """
    计算评估指标。使用 Huber Loss 报告, 避免共振峰离群值主导。
    """
    results = {}
    results["loss (test/val)"] = F.mse_loss(prediction, output).item()
    results["L1 Loss / (test/val)"] = F.l1_loss(prediction, output).item()

    if prediction.ndim >= 4 and prediction.shape[-1] == 2:
        pred_amp = torch.sqrt(prediction[..., 0]**2 + prediction[..., 1]**2 + 1e-8)
        out_amp = torch.sqrt(output[..., 0]**2 + output[..., 1]**2 + 1e-8)
        results["Amplitude MSE"] = F.mse_loss(pred_amp, out_amp).item()

    if verbose:
        for key, val in results.items():
            _log(f"{key} = {val:4.4}", logger)
        _wandb_log({key: val for key, val in results.items()}, logger, epoch)

    return results


def save_model(savepath, epoch, model, optimizer, loss, name="checkpoint_best"):
    os.makedirs(savepath, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, os.path.join(savepath, name))


def _log(msg, logger):
    """简易日志: 若 logger 可用则用 logger，否则 print"""
    if logger and hasattr(logger, 'info'):
        logger.info(msg)
    else:
        print(msg)


def _wandb_log(data, logger, epoch=None):
    """可选 wandb 日志"""
    if logger and hasattr(logger, 'log'):
        if epoch is not None:
            data['Epoch'] = epoch
        try:
            logger.log(data)
        except Exception:
            pass
