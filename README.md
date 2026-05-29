# Geometric FRF — 基于几何的频响函数预测

输入 3D 几何 → DeepONet-SIREN → 模态参数 (ω, ζ, φ) → 物理公式重建 FRF。

## 架构

```
(x,y,z) → Trunk (SIREN, w0=30) → φ_k(x) 模态振型
global_features → Branch (MLP) → pool → ω_k, ζ_k 频率+阻尼
PhysicsDecoder(无参数): φ + ω + ζ + freq → FRF(Re,Im)
```

## 数据

ANSYS MAPDL 生成 3D 固支板数据集:
- 铝板 100×60×10mm, 中央 45×8mm 夹紧区
- 前 2 阶模态, 质量归一化振型
- 40 频率点 (自适应网格, 共振峰附近密集)
- 300 样本 (train/val/test = 200/50/50)
- HDF5 per-sample-group 格式, 可变节点数 (~4k/样本)

## 损失

`MSE(ω)×100 + MSE(ζ)×1e5 + MSE(φ)×1` — 纯模态参数监督, FRF 由物理公式保证。

## 目录

```
geometric_frf/
├── models/        modal_model.py (PhysicsDecoder + ModalFRFModel)
│                  siren.py, frf_model.py, geometry_data.py
├── data/          dataset.py (HDF5 flat + per-sample-group)
├── training/      losses.py (modal_loss), trainer.py
├── ansys/         generate_3d_test.py (ANSYS数据生成)
└── sample/        run_validation.py (训练入口)
                   evaluate.py (评估), predict.py, 测试.py, 对比图.py
```

## 快速开始

```bash
# 1. 生成 ANSYS 数据 (需 ANSYS MAPDL license)
F:\pytorch_cuda12\python.exe ansys/generate_3d_test.py

# 2. 训练
F:\pytorch_cuda12\python.exe sample/run_validation.py

# 3. 评估 + 可视化
F:\pytorch_cuda12\python.exe sample/evaluate.py
F:\pytorch_cuda12\python.exe sample/对比图.py
```

## 配置

| 参数 | 值 |
|------|-----|
| 模型 | DeepONet-SIREN, 333K 参数 |
| hidden_dim | 256 |
| trunk_layers / branch_layers | 4 / 3 |
| n_modes | 2 |
| siren_w0 | 30 |
| lr | 0.0003, AdamW, CosineAnnealing |
| amp_scale | 500000 |
| freq_range | 1 ~ 8000 Hz |
