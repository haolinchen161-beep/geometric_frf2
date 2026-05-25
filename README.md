# Geometric FRF — 基于几何的模态参数预测

输入: 几何坐标 + 点特征 → DeepONet-SIREN → 模态参数 (ω, ζ, φ) → 物理重建 FRF。

## 架构

```
(x,y,z) → Trunk (SIREN) → φ_k(x) 模态振型
features → Branch (MLP) → pool → ω_k, ζ_k 频率+阻尼
PhysicsDecoder: φ + ω + ζ + freq → FRF(Re,Im) (无参数)
```

## 目录

```
geometric_frf/
├── models/      模态模型 + 编码器 + SIREN
├── data/        HDF5数据集
├── training/    损失 + 训练循环
├── configs/     配置文件
├── scripts/     CLI入口
└── sample/      数据生成 + 训练 + 评估 + 可视化
```

## 损失

`MSE(ω) + MSE(ζ) + MSE(φ)` — 纯参数监督，FRF由物理公式自动保证。

## 快速开始

```bash
F:\pytorch_cuda12\python.exe sample\generate_data.py
F:\pytorch_cuda12\python.exe sample\run_validation.py
F:\pytorch_cuda12\python.exe sample\evaluate.py
F:\pytorch_cuda12\python.exe sample\对比图.py
```

## 当前结果

| 指标 | 数值 |
|------|------|
| 模型 | DeepONet-SIREN, 335K参数 |
| 频率误差 | ~0.2 Hz (第一峰) |
| 阻尼比误差 | ~0.1% |
| 振型误差 | ~3% |
