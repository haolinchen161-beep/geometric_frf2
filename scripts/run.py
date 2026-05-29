"""
run.py — 几何→FRF 模型训练入口。

用法:
    F:\pytorch_cuda12\python.exe geometric_frf/scripts/run.py \
        --config configs/dataset.yaml \
        --model_cfg configs/pointnet_frf.yaml \
        --data_dir ./data
"""

import os
import sys
import argparse
import time

# 添加 geometric_frf 父目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import yaml

from models import build_geometric_model
from data import get_geometric_dataloader
from training import train, evaluate


def parse_args():
    parser = argparse.ArgumentParser(description='几何→FRF 训练')
    parser.add_argument('--config', default='configs/dataset.yaml', help='数据集配置文件')
    parser.add_argument('--model_cfg', default='configs/pointnet_frf.yaml', help='模型配置文件')
    parser.add_argument('--data_dir', default='./data', help='数据根目录')
    parser.add_argument('--dir', default='output', help='输出目录')
    parser.add_argument('--device', default='cuda', help='cuda 或 cpu')
    parser.add_argument('--fp16', type=lambda x: x == 'True', default=True, help='混合精度')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--continue_training', action='store_true', help='从检查点继续训练')
    return parser.parse_args()


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    print(args)

    # 加载配置
    config = load_yaml(args.config)
    model_cfg = load_yaml(args.model_cfg)

    # 合并学习率
    if 'lr' not in config:
        config.setdefault('optimizer', {}).setdefault('kwargs', {})['lr'] = model_cfg.get('lr', 0.001)

    # 输出目录
    args.dir = os.path.join(args.dir, time.strftime('%Y%m%d_%H%M%S'))

    # 构建 DataLoader
    trainloader, valloader, testloader, _, _, _ = get_geometric_dataloader(
        args, config, data_dir=args.data_dir,
    )

    # 构建模型
    net = build_geometric_model(
        model_cfg.get('encoder_type', 'pointnet'),
        model_cfg.get('encoder_kwargs', {}),
        model_cfg.get('decoder_kwargs', {}),
    ).to(args.device)

    # 打印模型信息
    total_params = sum(p.numel() for p in net.parameters())
    print(f"模型参数量: {total_params:,}")

    if args.debug:
        print("调试模式结束。")
        return

    # 构建优化器
    opt_cfg = config.get('optimizer', {})
    optimizer_class = getattr(torch.optim, opt_cfg.get('name', 'AdamW'))
    optimizer = optimizer_class(net.parameters(), **opt_cfg.get('kwargs', {'lr': 0.001}))

    # 学习率调度器 (Cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.get('epochs', 500), eta_min=1e-5
    )

    # 继续训练
    if args.continue_training:
        checkpoint_path = os.path.join(args.dir, "checkpoint_best")
        if os.path.exists(checkpoint_path):
            data = torch.load(checkpoint_path)
            net.load_state_dict(data["model_state_dict"])
            optimizer.load_state_dict(data["optimizer_state_dict"])
            print(f"继续训练，从 epoch {data['epoch']} 开始")

    # 训练
    net = train(args, config, model_cfg, net, trainloader, optimizer,
                valloader, scheduler)

    # 测试
    print("在测试集上评估...")
    results = evaluate(args, config, net, testloader, verbose=True)
    print(f"测试损失: {results['loss (asinh-MSE)']:4.4f}")


if __name__ == '__main__':
    main()
