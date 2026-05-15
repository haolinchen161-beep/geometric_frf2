"""
trainer.py — 逐点 FRF 训练循环 + 评估。

数据流:
    geometry + frequencies → net → per_point_frf (B, N, n_freqs)
    loss = MSE(prediction, target)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F


def train(args, config, model_cfg, net, dataloader, optimizer,
          valloader, scheduler, logger=None):
    """
    逐点 FRF 训练循环。

    参数:
        args:      CLI 参数 (含 device, fp16)
        config:    数据集配置 (含 epochs, optimizer 等)
        model_cfg: 模型配置
        net:       GeometricFRFModel 实例
    """
    lowest = np.inf
    net.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    for epoch in range(config.get('epochs', 500)):
        losses = []
        for batch in dataloader:
            optimizer.zero_grad()

            geometry = batch['geometry'].to(args.device)
            target = batch['point_frf'].to(args.device)
            frequencies = batch['frequencies'].to(args.device)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                prediction = net(geometry, frequencies)
                loss = F.mse_loss(prediction, target)

            losses.append(loss.detach().cpu().item())

            scaler.scale(loss).backward()

            grad_clip = config.get('optimizer', {}).get('gradient_clip')
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

        if scheduler is not None:
            scheduler.step(epoch)

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
    """计算 MSE 和 L1 损失"""
    results = {}
    results["loss (test/val)"] = F.mse_loss(prediction, output).item()
    results["L1 Loss / (test/val)"] = F.l1_loss(prediction, output).item()

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
