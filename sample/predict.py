"""
predict.py — 加载 checkpoint 对测试样本预测 FRF。
用法: F:\pytorch_cuda12\python.exe geometric_frf/sample/predict.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import torch, numpy as np
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


def main():
    testset = GeometricHDF5Dataset(['test.h5'], CONFIG, data_dir=data_dir, normalization=True, test=True)
    testset_raw = GeometricHDF5Dataset(['test.h5'], CONFIG, data_dir=data_dir, normalization=False, test=True)

    net = build_geometric_model(MODEL_CFG['encoder_type'], MODEL_CFG['encoder_kwargs'], MODEL_CFG['decoder_kwargs']).to(device)
    ckpt = torch.load(os.path.join(out_dir, "checkpoint_best"), map_location=device)
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()
    print(f"Checkpoint epoch={ckpt['epoch']}, loss={ckpt['loss']:.6f}, params={sum(p.numel() for p in net.parameters()):,}")

    all_preds, all_targets = [], []
    for idx in range(len(testset)):
        sn, sr = testset[idx], testset_raw[idx]
        gd = GeometryData(points=sn['geometry'].points.unsqueeze(0),
                          point_features=sn['geometry'].point_features.unsqueeze(0)
                          if sn['geometry'].point_features is not None else None).to(device)
        with torch.no_grad():
            frf_p, _, _, _ = net(gd, sn['frequencies'].unsqueeze(0).to(device))
        p = torch.clamp(frf_p.squeeze(0).cpu(), -5000, 5000)  # 已是物理空间
        t = testset_raw.undo_normalize(sn['point_frf'])       # asinh→物理
        all_preds.append(torch.sqrt(p[...,0]**2+p[...,1]**2+1e-8))
        all_targets.append(torch.sqrt(t[...,0]**2+t[...,1]**2+1e-8))

    all_preds = torch.stack(all_preds); all_targets = torch.stack(all_targets)
    mse = torch.nn.functional.mse_loss(all_preds, all_targets).item()
    print(f"测试集幅值MSE: {mse:.6f}")
    np.savez(os.path.join(out_dir, "predictions.npz"), predicted=all_preds.numpy(), target=all_targets.numpy())
    print(f"保存: {out_dir}/predictions.npz")


if __name__ == '__main__':
    main()
