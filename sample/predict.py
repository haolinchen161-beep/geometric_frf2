"""
predict.py — 使用训练好的模型进行预测。

加载训练好的检查点, 对测试样本预测逐点FRF, 并输出结果。

用法:
    F:\pytorch_cuda12\python.exe geometric_frf/sample/predict.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import torch, numpy as np

from geometric_frf.models import build_geometric_model, GeometryData
from geometric_frf.data.dataset import GeometricHDF5Dataset

# 复用 run_validation.py 的配置
CONFIG = {
    'freq_sample': False,
    'freq_limit': 90,
    'n_freqs': 120,
    'max_frequency': None,
    'resonance_weight': 2.0,
    'amp_weight': 0.0,
    'data_path_train': ['train.h5'],
    'data_path_val': ['val.h5'],
    'data_paths_test': ['test.h5'],
}
MODEL_CFG = {
    'encoder_type': 'gnn',
    'encoder_kwargs': {
        'in_channels': 11, 'hidden_dim': 256, 'out_dim': 256,
        'coord_dim': 3, 'n_layers': 4, 'conv_type': 'sage',
        'use_global_pool': True, 'global_pool': 'mean',
    },
    'decoder_kwargs': {
        'in_dim': 256, 'n_freqs': 120, 'hidden_dim': 256,
        'n_layers': 4, 'chunk_size': 256, 'out_dim': 1,
        'freq_encoding': 'sin',
    },
}


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    ckpt_path = os.path.join(os.path.dirname(__file__), "output", "checkpoint_best")

    print("=" * 60)
    print("模型预测演示")
    print("=" * 60)

    # ---- 加载数据 ----
    testset = GeometricHDF5Dataset(
        ['test.h5'], CONFIG, data_dir=data_dir, normalization=True, test=True
    )
    print(f"测试集: {len(testset)} 样本")

    # ---- 加载模型 ----
    net = build_geometric_model(
        MODEL_CFG['encoder_type'],
        MODEL_CFG['encoder_kwargs'],
        MODEL_CFG['decoder_kwargs'],
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()
    print(f"模型加载: epoch={ckpt['epoch']}, checkpoint_loss={ckpt['loss']:.4f}")

    # ---- 预测前5个样本 ----
    print("\n" + "-" * 40)
    print("逐样本预测结果")
    print("-" * 40)

    for idx in range(min(5, len(testset))):
        sample = testset[idx]
        # 单样本 → 加 batch 维度 (N,*) → (1, N, *)
        gd = GeometryData(
            points=sample['geometry'].points.unsqueeze(0),
            point_features=(sample['geometry'].point_features.unsqueeze(0)
                            if sample['geometry'].point_features is not None else None),
            edge_index=(sample['geometry'].edge_index.unsqueeze(0)
                        if sample['geometry'].edge_index is not None else None),
            batch=None,
        ).to(device)
        freq = sample['frequencies'].unsqueeze(0).to(device)  # (1, n_freqs)
        target = sample['point_frf']  # (N, n_freqs)

        with torch.no_grad():
            pred_asinh = net(gd, freq).squeeze(0).cpu()  # (N, n_freqs) asinh空间

        # 逆变换 → 真实物理 FRF
        pred_physical = testset.undo_normalize(pred_asinh)       # torch.sinh()
        target_physical = testset.undo_normalize(target)         # torch.sinh()

        # 物理空间 MSE
        mse = torch.nn.functional.mse_loss(pred_physical, target_physical)

        print(f"\n样本 {idx}:")
        print(f"  预测形状: {pred_physical.shape} (N={pred_physical.shape[0]}, F={pred_physical.shape[1]})")
        print(f"  物理FRF MSE: {mse.item():.6f}")
        print(f"  预测范围: [{pred_physical.min().item():.4f}, {pred_physical.max().item():.4f}]")
        print(f"  目标范围: [{target_physical.min().item():.4f}, {target_physical.max().item():.4f}]")

        # 前5个点的FRF值 (前10个频率)
        print(f"  点#0  预测FRF前10频率: {pred_physical[0, :10].tolist()}")
        print(f"  点#0  目标FRF前10频率: {target_physical[0, :10].tolist()}")

    # ---- 保存预测结果 ----
    print("\n" + "-" * 40)
    print("保存预测结果...")
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)

    all_preds, all_targets = [], []
    for idx in range(len(testset)):
        sample = testset[idx]
        gd = GeometryData(
            points=sample['geometry'].points.unsqueeze(0),
            point_features=(sample['geometry'].point_features.unsqueeze(0)
                            if sample['geometry'].point_features is not None else None),
        ).to(device)
        freq = sample['frequencies'].unsqueeze(0).to(device)
        with torch.no_grad():
            pred_asinh = net(gd, freq).squeeze(0).cpu()
        # 保存物理值
        all_preds.append(testset.undo_normalize(pred_asinh))
        all_targets.append(testset.undo_normalize(sample['point_frf']))

    np.savez(os.path.join(out_dir, "predictions.npz"),
             predictions=torch.stack(all_preds).numpy(),
             targets=torch.stack(all_targets).numpy())
    print(f"预测结果保存到: {out_dir}/predictions.npz")
    print(f"  predictions: {torch.stack(all_preds).shape}")
    print(f"  targets:     {torch.stack(all_targets).shape}")

    # 整体统计
    preds_t = torch.stack(all_preds)
    targs_t = torch.stack(all_targets)
    overall_mse = torch.nn.functional.mse_loss(preds_t, targs_t).item()
    print(f"\n整体测试MSE: {overall_mse:.4f}")
    print("完成!")


if __name__ == '__main__':
    main()
