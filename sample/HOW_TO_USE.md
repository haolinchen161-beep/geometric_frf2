# geometric_frf 使用说明 — ANSYS 3D 固支板

## 1. 目录结构

```
geometric_frf/
├── models/
│   ├── modal_model.py        DeepONet-SIREN (PhysicsDecoder + ModalFRFModel)
│   ├── frf_model.py          模型工厂 build_geometric_model()
│   ├── geometry_data.py      GeometryData 数据容器
│   └── siren.py              SIREN 正弦激活 (w0=30)
├── data/
│   └── dataset.py            HDF5 数据集 (flat + per-sample-group) + collate
├── training/
│   ├── losses.py             modal_loss — ω×100 + ζ×1e5 + φ×1
│   └── trainer.py            训练循环 + 评估 (支持可变N/F)
├── ansys/
│   ├── generate_3d_test.py   ANSYS MAPDL 数据生成
│   ├── data/                 train/val/test.h5
│   └── mesh_viz/             网格截图
└── sample/
    ├── run_validation.py     训练入口
    ├── evaluate.py           评估 + 保存 final_results.npz
    ├── 测试.py               查看原始 FRF (峰值+Re/Im)
    ├── 对比图.py             预测 vs 真实 (幅值+实部+虚部)
    ├── predict.py            推理
    └── output/               checkpoint + 图表 + npz
```

## 2. 数据

### ANSYS 3D 固支板

| 参数 | 值 |
|------|-----|
| 板尺寸 | 100×60×10mm (铝, E=71.7GPa, ρ=2810) |
| 固定方式 | 上下表面中央 45×8mm 区域全约束 |
| 激励点 | (20%L, 20%W, H) 顶面自由角 |
| 模态 | 前 2 阶, 质量归一化振型 (ΦᵀMΦ=I) |
| 频率网格 | 40 点, 自适应 (共振峰 ±3·半功率带宽密集) |
| 样本 | 300 (200/50/50), 几何±10% 随机化 |
| 网格 | 6mm 自由四面体, ~4k 节点/样本 |

### HDF5 格式 (per-sample-group)

```
/sample_N/
├── points        (N_nodes, 3)     节点坐标
├── point_frf     (N_nodes, F, 2)  复数 FRF [Re, Im]
├── frequencies   (F,)             频率 Hz
├── point_features (6,)            全局特征 (E,ρ,L,W,H 归一化 + n_modes)
├── modal_omega   (2,)             固有圆频率 rad/s
├── modal_zeta    (2,)             阻尼比 (=0.003)
├── modal_phi     (N_nodes, 2)     质量归一化振型
└── modal_phi_exc (2,)             激励点振型值 φ_k(x_f)
```

### FRF 公式 (物理正确)

```
H(x, x_f, ω) = Σ_k φ_k(x)·φ_k(x_f) / (ω_k² - ω² + j·2ζ_k·ω_k·ω)
```

### 归一化

- FRF: `torch.asinh()` → 评估时 `torch.sinh()` 还原
- 频率: `(f - 1) / (8000 - 1) × 2 - 1` → 映射到 [-1, 1]

## 3. 模型

```
Trunk (SIREN):  (x,y,z) → SirenMLP(w0=30) → φ_k(x)  模态振型
Branch (MLP):   point_features → MLP → pool → ω_k, ζ_k
PhysicsDecoder(无参数): φ + ω + ζ + φ_exc + freq → FRF(Re,Im)
```

## 4. 使用

```bash
# 生成数据 (需 ANSYS MAPDL license, ~数小时)
F:\pytorch_cuda12\python.exe ansys/generate_3d_test.py

# 查看原始 FRF
F:\pytorch_cuda12\python.exe sample/测试.py

# 训练
F:\pytorch_cuda12\python.exe sample/run_validation.py

# 评估
F:\pytorch_cuda12\python.exe sample/evaluate.py

# 对比图
F:\pytorch_cuda12\python.exe sample/对比图.py
```

## 5. 当前配置

| 参数 | 值 |
|------|-----|
| 模型 | DeepONet-SIREN, 333K 参数 |
| hidden_dim / trunk / branch | 256 / 4 层 / 3 层 |
| n_modes | 2 |
| siren_w0 | 30 |
| amp_scale | 500000 |
| freq_range | 1 ~ 8000 Hz |
| 损失 | ω×100 + ζ×1e5 + φ×1 |
| 优化器 | AdamW, lr=0.0003, CosineAnnealing |
| batch_size | 4 (3D 板节点多) |
| FP16 | 关闭 (SIREN 易 NaN) |
| 训练 | 200 × 2000 epochs |
