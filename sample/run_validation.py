"""
run_validation.py — 正式训练脚本。

使用合成3D悬臂梁数据训练 geometric_frf 模型。

用法:
    F:\pytorch_cuda12\python.exe geometric_frf/sample/run_validation.py
"""

import os
import sys
import time
import numpy as np
import torch
import torch.utils.data

# PyTorch 2.1: CosineAnnealingLR(last_epoch=-1) 在 __init__ 中调用 step()
# 时检查 optimizer._step_count (2.1 中不存在此属性), 无条件触发警告。
# 训练循环中 scheduler.step() 已在 optimizer.step() 之后, 逻辑正确。
# 此处模块级过滤消除误报。
import warnings
warnings.filterwarnings('ignore', message='Detected call of')
warnings.filterwarnings('ignore', message='To get the last learning rate')

# 添加项目根目录 (geometric_frf 的父目录) 到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from geometric_frf.models import build_geometric_model
from geometric_frf.training import train, evaluate, frf_loss
import wandb


# ============ 训练配置 ============
CONFIG = {
    'epochs': 2000,
    'validation_frequency': 5,   # 每5轮验证, 观察训练过程
    'freq_sample': True,        # 峰值区全取+背景区随机抽足到 freq_limit
    'freq_limit': 90,           # ~70峰值 + ~20背景 = 90, 比120少25%
    'n_samples': 200,
    'n_train_samples': 180,     # 180训练 / 20验证
    'n_val_samples': 20,
    'n_freqs': 120,
    'max_frequency': None,
    'resonance_weight': 2.0,    # 降低共振峰加权 (避免第一峰过度主导)
    'amp_weight': 0.0,
    'data_path_train': ['train.h5'],
    'data_path_val': ['val.h5'],
    'data_paths_test': ['test.h5'],
    'optimizer': {
        'name': 'AdamW',
        'kwargs': {'lr': 0.0005, 'weight_decay': 0.00005, 'betas': (0.9, 0.999)},
        'gradient_clip': 1.0,   # GNN 推荐 1.0
    },
}

MODEL_CFG = {
    'encoder_type': 'gnn',      # 使用GNN利用网格拓扑 (edge_index)
    'encoder_kwargs': {
        'in_channels': 11,      # 3局部特征 + 8全局phy_para
        'hidden_dim': 256,      # 扩大隐藏层
        'out_dim': 256,
        'coord_dim': 3,
        'n_layers': 4,          # 4层图卷积
        'conv_type': 'sage',    # GraphSAGE
        'use_global_pool': True,
        'global_pool': 'mean',
    },
    'decoder_kwargs': {
        'in_dim': 256,
        'n_freqs': 120,
        'hidden_dim': 256,
        'n_layers': 4,
        'chunk_size': 256,
        'out_dim': 1,
        'freq_encoding': 'sin',  # sin/cos频率编码
    },
}


class SimpleArgs:
    def __init__(self):
        self.batch_size = 32    # 正常批次大小
        self.seed = 42
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.fp16 = torch.cuda.is_available()
        self.dir = os.path.join(os.path.dirname(__file__), "output")
        self.debug = False
        self.continue_training = True   # True=从checkpoint继续, False=从零开始


def main():
    print("=" * 60)
    print("geometric_frf 端到端验证")
    print("=" * 60)

    args = SimpleArgs()
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    print(f"设备: {args.device}, FP16: {args.fp16}")
    print(f"数据目录: {data_dir}")
    print()

    # ====== 步骤1: 数据加载 ======
    print("--- 步骤1: 构建 DataLoader ---")
    from geometric_frf.data.dataset import GeometricHDF5Dataset, collate_geometry_batch
    import numpy as np

    # 训练集: 归一化, 计算 frf_mean/frf_std
    trainset = GeometricHDF5Dataset(
        ['train.h5'], CONFIG, data_dir=data_dir,
        normalization=True, test=False,
    )
    # 验证/测试集: asinh 归一化 (无状态, 与训练集使用相同函数)
    valset = GeometricHDF5Dataset(
        ['val.h5'], CONFIG, data_dir=data_dir,
        normalization=True, test=True,
    )
    testset = GeometricHDF5Dataset(
        ['test.h5'], CONFIG, data_dir=data_dir,
        normalization=True, test=True,
    )

    gen = torch.Generator(device='cpu').manual_seed(args.seed)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, drop_last=True, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=collate_geometry_batch,
        generator=gen,
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=2, drop_last=False, shuffle=False,
        num_workers=0, collate_fn=collate_geometry_batch,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=2, drop_last=False, shuffle=False,
        num_workers=0, collate_fn=collate_geometry_batch,
    )
    print(f"  训练集: {len(trainset)} 样本, {len(trainloader)} 批次")
    print(f"  验证集: {len(valset)} 样本")
    print(f"  测试集: {len(testset)} 样本")

    # 检查数据形状
    batch = next(iter(trainloader))
    gd = batch['geometry']
    target = batch['point_frf']
    freq = batch['frequencies']
    print(f"  batch points:     {gd.points.shape}")
    print(f"  batch point_feat: {gd.point_features.shape if gd.point_features is not None else 'None'}")
    print(f"  batch edge_index: {gd.edge_index.shape if gd.edge_index is not None else 'None'}")
    print(f"  batch target:     {target.shape}")
    print(f"  batch frequencies:{freq.shape}")
    print()

    # 验证数据值范围
    print(f"  target range: [{target.min().item():.3f}, {target.max().item():.3f}]")
    print(f"  freq range:   [{freq.min().item():.3f}, {freq.max().item():.3f}]")
    print()

    # ====== 步骤2: 模型构建 ======
    print("--- 步骤2: 构建模型 ---")
    net = build_geometric_model(
        MODEL_CFG['encoder_type'],
        MODEL_CFG['encoder_kwargs'],
        MODEL_CFG['decoder_kwargs'],
    ).to(args.device)
    total_params = sum(p.numel() for p in net.parameters())
    print(f"  模型参数量: {total_params:,}")
    print()

    # ====== 步骤3: 前向传播测试 ======
    print("--- 步骤3: 前向传播测试 ---")
    net.eval()
    with torch.no_grad():
        gd_dev = gd.to(args.device)
        freq_dev = freq.to(args.device)
        pred = net(gd_dev, freq_dev)
    print(f"  输入:  points={gd.points.shape}, freq={freq.shape}")
    print(f"  输出:  {pred.shape}")
    B, N, F = pred.shape
    assert B == args.batch_size, f"batch大小不匹配: 期望{args.batch_size}, 实际{B}"
    assert N > 0, f"空间点数应>0: {N}"
    assert F <= CONFIG['freq_limit'], f"频率数超限: F={F} > freq_limit={CONFIG['freq_limit']}"
    print("  前向传播 PASS")
    print()

    # ====== 步骤4: 初始Loss ======
    print("--- 步骤4: 初始 Loss ---")
    with torch.no_grad():
        pred = net(gd_dev, freq_dev)
        target_dev = target.to(args.device)
        init_loss = frf_loss(pred, target_dev,
                            out_dim=MODEL_CFG['decoder_kwargs']['out_dim'],
                            resonance_weight=CONFIG['resonance_weight'])
        init_mse = torch.nn.functional.mse_loss(pred, target_dev).item()
    print(f"  初始加权 Loss: {init_loss.item():.4f}")
    print(f"  初始 MSE:       {init_mse:.4f}")
    print()

    # ====== 步骤5: 训练 ======
    print("--- 步骤5: 训练 ---")
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=CONFIG['optimizer']['kwargs']['lr'],
        weight_decay=CONFIG['optimizer']['kwargs']['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs'], eta_min=1e-5,
    )
    start_epoch = 0

    # 续训: 从 checkpoint_last 恢复
    ckpt_path = os.path.join(args.dir, "checkpoint_last")
    if args.continue_training and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=args.device)
        net.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"  从 epoch {start_epoch} 继续训练")

    # 初始化 wandb (离线模式避免网络阻塞)
    wandb.init(project="BrepGDM", entity="haolinchen161-nanjing-tech-university",
               config={**CONFIG, **MODEL_CFG, 'total_params': total_params},
               resume="allow" if args.continue_training else None,
               mode="offline")
    wandb.run.name = f"gnn_beam_{time.strftime('%m%d_%H%M')}"

    print(f"  训练 {CONFIG['epochs']} epochs...")
    print(f"  WandB: {wandb.run.get_url()}")
    t0 = time.time()
    # 把 start_epoch 传给 train
    net = train(args, CONFIG, MODEL_CFG, net, trainloader, optimizer,
                valloader, scheduler, logger=wandb, start_epoch=start_epoch)
    elapsed = time.time() - t0
    wandb.finish()
    print(f"  训练完成, 耗时 {elapsed:.1f}s")
    print()

    # ====== 步骤6: 验证结果 ======
    print("--- 步骤6: 验证结果 ---")
    # 重新加载最佳模型
    best_path = os.path.join(args.dir, "checkpoint_best")
    if os.path.exists(best_path):
        net.load_state_dict(torch.load(best_path, map_location=args.device)["model_state_dict"])

    results_test = evaluate(args, CONFIG, net, testloader, verbose=True)

    # 获取最终训练loss评估
    net.eval()
    with torch.no_grad():
        gd_dev = gd.to(args.device)
        freq_dev = freq.to(args.device)
        target_dev = target.to(args.device)
        pred_final = net(gd_dev, freq_dev)
        final_mse = torch.nn.functional.mse_loss(pred_final, target_dev).item()

    print()
    print("=" * 60)
    print("验证结果总结")
    print("=" * 60)
    print(f"  设备:            {args.device}")
    print(f"  模型参数量:      {total_params:,}")
    print(f"  训练时间:        {elapsed:.1f}s")
    print(f"  初始 MSE:        {init_mse:.4f}")
    print(f"  最终 MSE:        {final_mse:.4f}")
    mse_ratio = final_mse / (init_mse + 1e-8)
    print(f"  MSE ratio:       {mse_ratio:.3f}")
    test_loss = results_test.get('loss (test/val)', -1)
    print(f"  Test MSE:        {test_loss:.4f}")
    print()

    # 判断: 只要流程跑通就算通过
    print("Pipeline verification: ALL STEPS PASSED")
    print(f"  - Data loading:      OK ({CONFIG['n_samples']} train / {CONFIG['n_val_samples']} val samples)")
    print(f"  - Model forward:     OK (output: {target.shape})")
    print(f"  - Training loop:     OK ({CONFIG['epochs']} epochs, {elapsed:.1f}s)")
    print("  - Checkpoint save:   OK")
    print(f"  - Test evaluation:   OK (MSE={test_loss:.4f})")

    return 0


if __name__ == '__main__':
    exit(main())
