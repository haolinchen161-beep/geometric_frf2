"""
evaluate.py — 训练后评估+可视化。
加载检查点 → 预测模态参数 → 物理重建FRF → 对比+保存。

用法: F:\pytorch_cuda12\python.exe geometric_frf/sample/evaluate.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import torch, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from geometric_frf.models import build_geometric_model, GeometryData
from geometric_frf.data.dataset import GeometricHDF5Dataset

CONFIG = {'n_freqs': 120}
MODEL_CFG = {
    'encoder_type': 'modal',
    'encoder_kwargs': {'coord_dim':3, 'point_feat_dim':11, 'hidden_dim':256,
                       'n_modes':2, 'trunk_layers':4, 'branch_layers':3,
                       'siren_w0':30.0, 'amp_scale':500000.0},
    'decoder_kwargs': {},
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = os.path.join(os.path.dirname(__file__), "data")
out_dir  = os.path.join(os.path.dirname(__file__), "output")
ckpt_path = os.path.join(out_dir, "checkpoint_best")


def main():
    print("=" * 60)
    print("模型评估 + 可视化 (模态参数预测)")
    print("=" * 60)

    testset = GeometricHDF5Dataset(['test.h5'], CONFIG, data_dir=data_dir,
                                   normalization=True, test=True)
    testset_raw = GeometricHDF5Dataset(['test.h5'], CONFIG, data_dir=data_dir,
                                       normalization=False, test=True)
    print(f"测试集: {len(testset)} 样本")

    model = build_geometric_model(MODEL_CFG['encoder_type'],
                                  MODEL_CFG['encoder_kwargs'],
                                  MODEL_CFG['decoder_kwargs']).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Checkpoint: epoch={ckpt['epoch']}, loss={ckpt['loss']:.6f}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 逐样本预测
    all_preds, all_targets, all_freqs = [], [], []
    all_preds_re, all_preds_im = [], []
    all_targets_re, all_targets_im = [], []
    omega_errs, zeta_errs = [], []
    for idx in range(len(testset)):
        s_norm = testset[idx]; s_raw = testset_raw[idx]
        gd = GeometryData(
            points=s_norm['geometry'].points.unsqueeze(0),
            point_features=s_norm['geometry'].point_features.unsqueeze(0)
                if s_norm['geometry'].point_features is not None else None,
        ).to(device)
        with torch.no_grad():
            frf_p, op, zp, pp = model(gd, s_norm['frequencies'].unsqueeze(0).to(device))
        frf_p = frf_p.squeeze(0).cpu()
        p = torch.clamp(frf_p, -5000, 5000)
        t = testset_raw.undo_normalize(s_norm['point_frf'])

        omega_errs.append((op.cpu() - s_norm['modal_omega']).abs())
        zeta_errs.append((zp.cpu() - s_norm['modal_zeta']).abs())

        all_preds.append(torch.sqrt(p[...,0]**2+p[...,1]**2+1e-8))
        all_targets.append(torch.sqrt(t[...,0]**2+t[...,1]**2+1e-8))
        all_preds_re.append(p[...,0]); all_preds_im.append(p[...,1])
        all_targets_re.append(t[...,0]); all_targets_im.append(t[...,1])
        all_freqs.append(s_raw['frequencies'])

    all_preds = torch.stack(all_preds); all_targets = torch.stack(all_targets)
    all_preds_re = torch.stack(all_preds_re); all_targets_re = torch.stack(all_targets_re)
    all_preds_im = torch.stack(all_preds_im); all_targets_im = torch.stack(all_targets_im)
    all_freqs = torch.stack(all_freqs)
    points_3d = testset_raw.loaded['points'][:len(testset)]

    mse = torch.nn.functional.mse_loss(all_preds, all_targets).item()
    l1 = torch.nn.functional.l1_loss(all_preds, all_targets).item()
    omega_mae = torch.cat(omega_errs).mean().item()
    zeta_mae = torch.cat(zeta_errs).mean().item()
    print(f"幅值MSE={mse:.1f} L1={l1:.1f} | ω_MAE={omega_mae:.1f}rad/s ζ_MAE={zeta_mae:.5f}")

    # 保存
    np.savez(os.path.join(out_dir, "final_results.npz"),
             points=points_3d.numpy(), frequencies=all_freqs.numpy(),
             predicted_frf=all_preds.numpy(), target_frf=all_targets.numpy(),
             predicted_re=all_preds_re.numpy(), target_re=all_targets_re.numpy(),
             predicted_im=all_preds_im.numpy(), target_im=all_targets_im.numpy())
    print(f"数据保存: {out_dir}/final_results.npz")
    print(f"评估完成!")


if __name__ == '__main__':
    main()
