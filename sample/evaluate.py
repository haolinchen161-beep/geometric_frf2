"""
evaluate.py — 训练后完整评估 + 可视化。

步骤:
    1. 加载最佳 checkpoint
    2. 对所有测试样本预测 → 逆变换为物理 FRF
    3. 画图验证 (多点 FRF 曲线 + 空间分布)
    4. 保存逐点 FRF 到文件

用法:
    F:\pytorch_cuda12\python.exe geometric_frf/sample/evaluate.py
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import torch, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from geometric_frf.models import build_geometric_model, GeometryData
from geometric_frf.data.dataset import GeometricHDF5Dataset

# ============ 配置 ============
CONFIG = {
    'freq_sample': False, 'freq_limit': 72, 'n_freqs': 120,
    'data_path_train': ['train.h5'], 'data_path_val': ['val.h5'],
    'data_paths_test': ['test.h5'],
}
MODEL_CFG = {
    'encoder_type': 'gnn',
    'encoder_kwargs': {'in_channels':11, 'hidden_dim':256, 'out_dim':256,
                       'coord_dim':3, 'n_layers':4, 'conv_type':'sage',
                       'use_global_pool':True, 'global_pool':'mean'},
    'decoder_kwargs': {'in_dim':256, 'n_freqs':120, 'hidden_dim':256,
                       'n_layers':4, 'chunk_size':256, 'out_dim':1,
                       'freq_encoding':'sin'},
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = os.path.join(os.path.dirname(__file__), "data")
out_dir  = os.path.join(os.path.dirname(__file__), "output")
ckpt_path = os.path.join(out_dir, "checkpoint_best")

def main():
    print("=" * 60)
    print("模型评估 + 可视化")
    print("=" * 60)

    # ---- 1. 加载数据 ----
    testset = GeometricHDF5Dataset(
        ['test.h5'], CONFIG, data_dir=data_dir, normalization=True, test=True)
    # 原始数据 (用于频率轴)
    testset_raw = GeometricHDF5Dataset(
        ['test.h5'], CONFIG, data_dir=data_dir, normalization=False, test=True)
    print(f"测试集: {len(testset)} 样本")

    # ---- 2. 加载模型 ----
    model = build_geometric_model(
        MODEL_CFG['encoder_type'], MODEL_CFG['encoder_kwargs'],
        MODEL_CFG['decoder_kwargs']).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Checkpoint: epoch={ckpt['epoch']}, loss={ckpt['loss']:.6f}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,}")

    # ---- 3. 逐样本预测 ----
    # 直接保存各样本原始120点自适应数据, 不做插值 (插值会丢失峰值精度)
    all_preds, all_targets, all_freqs = [], [], []
    for idx in range(len(testset)):
        s_norm = testset[idx]
        s_raw  = testset_raw[idx]

        gd = GeometryData(
            points=s_norm['geometry'].points.unsqueeze(0),
            point_features=s_norm['geometry'].point_features.unsqueeze(0) \
                if s_norm['geometry'].point_features is not None else None,
        ).to(device)

        freq_norm = s_norm['frequencies'].unsqueeze(0).to(device)

        with torch.no_grad():
            pred_asinh = model(gd, freq_norm).squeeze(0).cpu()

        # 逆变换 → 物理 FRF (保留原始120点)
        pred_phys = testset.undo_normalize(pred_asinh)
        targ_phys = testset.undo_normalize(s_norm['point_frf'])
        freq_phys = s_raw['frequencies']  # (120,) 该样本自适应频率

        all_preds.append(pred_phys)
        all_targets.append(targ_phys)
        all_freqs.append(freq_phys)

    all_preds   = torch.stack(all_preds)   # (50, 240, 120)
    all_targets = torch.stack(all_targets) # (50, 240, 120)
    all_freqs   = torch.stack(all_freqs)   # (50, 120)  每样本独立频率轴
    points_3d   = testset_raw.loaded['points'][:len(testset)]  # (50, 240, 3)
    print(f"预测完成: {all_preds.shape} (原始120点, 无插值)")

    # ---- 4. 整体误差 ----
    mse = torch.nn.functional.mse_loss(all_preds, all_targets).item()
    l1  = torch.nn.functional.l1_loss(all_preds, all_targets).item()
    print(f"\n整体误差:  MSE={mse:.6f},  L1={l1:.6f}")

    # ---- 5. 可视化: 第0个样本, 选代表性点画 FRF ----
    sample_idx = 0
    pts = points_3d[sample_idx]  # (240, 3)
    target = all_targets[sample_idx]  # (240, 120)
    pred   = all_preds[sample_idx]    # (240, 120)
    f_np   = all_freqs[sample_idx].numpy()  # (120,) 该样本自适应频率

    # 沿梁长选 5 个点
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 图1: 多点 FRF (非线性拉伸x轴, 和测试.py/对比图.py一致)
    ax = axes[0, 0]
    # 从target检测两峰
    f_abs = np.abs(target.numpy()).max(axis=0)
    f_peaks = []
    for i in range(2, len(f_abs)-2):
        if (f_abs[i] > f_abs[i-1] and f_abs[i] > f_abs[i+1] and
            f_abs[i] > f_abs[i-2] and f_abs[i] > f_abs[i+2] and
            f_abs[i] > f_abs.max() * 0.015):
            f_peaks.append(f_np[i])
    f_peaks = sorted(f_peaks)[:2]
    pk1, pk2 = f_peaks[0], f_peaks[1]
    # 高斯权重拉伸
    bw1, bw2 = 5.0, 8.0
    w1 = 30.0 * np.exp(-0.5 * ((f_np - pk1) / (bw1 * 0.5))**2)
    w2 =  6.0 * np.exp(-0.5 * ((f_np - pk2) / (bw2 * 0.5))**2)
    tw = 1.0 + w1 + w2
    xs = np.zeros_like(f_np)
    for i in range(1, len(f_np)):
        df = f_np[i] - f_np[i-1]
        xs[i] = xs[i-1] + (tw[i] + tw[i-1]) / 2 * df

    x_positions = [0, 0.125, 0.25, 0.375, 0.5]
    colors = plt.cm.viridis(np.linspace(0, 1, 5))
    for x_targ, c in zip(x_positions, colors):
        idx = torch.argmin((pts[:,0]-x_targ).abs()).item()
        ax.plot(xs, target[idx].numpy(), color=c, alpha=0.4, linewidth=1)
        ax.plot(xs, pred[idx].numpy(), color=c, linewidth=1.5, linestyle='--',
                label=f'x={pts[idx,0]:.3f}')
    # 背景遮罩
    s1 = [np.interp(pk1-bw1, f_np, xs), np.interp(pk1+bw1, f_np, xs)]
    s2 = [np.interp(pk2-bw2, f_np, xs), np.interp(pk2+bw2, f_np, xs)]
    ax.axvspan(s1[0], s1[1], color='gray', alpha=0.08)
    ax.axvspan(s2[0], s2[1], color='gray', alpha=0.08)
    # 自定义刻度
    tfs = np.unique(np.sort([f_np[0], pk1-bw1, pk1, pk1+bw1, pk2-bw2, pk2, pk2+bw2, f_np[-1]]))
    tfs = tfs[(tfs >= f_np[0]) & (tfs <= f_np[-1])]
    tls = np.interp(tfs, f_np, xs)
    ax.set_xticks(tls)
    ax.set_xticklabels([f'{f:.0f}' for f in tfs], fontsize=7)
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('FRF (signed displacement)')
    ax.set_title('FRF at 5 Points Along Beam (solid=target, dashed=pred)')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # 图2: 散点图 (预测 vs 目标)
    ax = axes[0, 1]
    ax.scatter(target.flatten()[:5000], pred.flatten()[:5000], s=1, alpha=0.3)
    ax.plot([target.min(), target.max()], [target.min(), target.max()], 'r--', linewidth=1)
    ax.set_xlabel('Target FRF'); ax.set_ylabel('Predicted FRF')
    ax.set_title(f'Prediction vs Target (MSE={mse:.6f})')
    ax.grid(True, alpha=0.3)

    # 图3: 空间分布 (共振频率处)
    ax = axes[1, 0]
    # 找全局幅值最大的频率
    peak_f = torch.argmax(target.abs().max(dim=0)[0]).item()
    resp_targ = target[:, peak_f].numpy()
    resp_pred = pred[:, peak_f].numpy()
    ax.plot(pts[:, 0], resp_targ, 'b-', alpha=0.5, linewidth=1, label='Target')
    ax.plot(pts[:, 0], resp_pred, 'r--', linewidth=1.5, label='Predicted')
    ax.set_xlabel('X coordinate (m)'); ax.set_ylabel('FRF')
    ax.set_title(f'Spatial Pattern @ {f_np[peak_f]:.1f} Hz')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 图4: 误差分布直方图
    ax = axes[1, 1]
    errors = (pred - target).flatten().numpy()
    ax.hist(errors, bins=100, alpha=0.7, density=True)
    ax.axvline(x=0, color='r', linestyle='--')
    ax.set_xlabel('Prediction Error'); ax.set_ylabel('Density')
    ax.set_title(f'Error Distribution (std={errors.std():.4f})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, "evaluation.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表保存: {fig_path}")

    # ---- 6. 保存逐点 FRF ----
    npz_path = os.path.join(out_dir, "final_results.npz")
    np.savez(npz_path,
             points=points_3d.numpy(),           # (50, 240, 3)
             frequencies=all_freqs.numpy(),      # (50, 120) 每样本独立频率轴
             predicted_frf=all_preds.numpy(),    # (50, 240, 120)
             target_frf=all_targets.numpy(),     # (50, 240, 120)
    )
    print(f"数据保存: {npz_path}")
    print(f"\n使用方式:")
    print(f"  data = np.load('{npz_path}')")
    print(f"  data['points'][0, i]     # 第0个样本第i点的 (x,y,z) 坐标")
    print(f"  data['predicted_frf'][0, i]  # 该点的预测FRF曲线")
    print(f"  data['frequencies']      # 频率轴 (Hz)")
    print(f"\n完成!")

if __name__ == '__main__':
    main()
