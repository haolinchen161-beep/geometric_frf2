# Geometric FRF — 基于几何输入的逐点频响函数预测

输入：三维几何属性（点坐标、网格拓扑、FEM 网格等）  
输出：**每个点的完整频响函数 FRF** `(N_points, n_freqs)`

## 目录结构

```
geometric_frf/
├── models/
│   ├── film.py               FiLM 条件调制层
│   ├── geometry_data.py      GeometryData 统一数据容器
│   ├── geometric_encoder.py  编码器 (Simple/PointNet/GNN/DeepONet)
│   ├── perpoint_decoder.py   逐点FRF解码器 (Concat + sin/cos频率编码)
│   ├── frf_model.py          完整模型 + build_geometric_model()
│   └── siren.py              SIREN 正弦激活 (DeepONet用)
├── data/
│   └── dataset.py            HDF5数据集 (asinh归一化) + collate (GNN兼容)
├── training/
│   ├── losses.py             Huber Loss + 共振峰加权 + 复数FRF约束
│   └── trainer.py            训练循环 + 评估
├── configs/
│   ├── dataset.yaml / pointnet_frf.yaml / gnn_frf.yaml / deeponet_frf.yaml
├── scripts/
│   └── run.py                标准训练入口
└── sample/                   完整验证示例
    ├── generate_data.py      3D悬臂梁数据生成 (Euler-Bernoulli物理模型)
    ├── run_validation.py     训练脚本
    ├── evaluate.py           训练后评估 + 可视化
    ├── 测试.py               查看真实FRF数据
    ├── predict.py            预测/推理
    └── output/               输出 (checkpoint + 图表 + 结果)
```

## 数据流

```
HDF5 → Dataset (asinh压缩) → GeometryData (点坐标+特征+拓扑)
     → Encoder (GNN/PointNet/DeepONet) → 逐点隐特征 (B, N, D)
     → Decoder (Concat+sin编码) + 频率 → 逐点FRF (B, N, n_freqs)
     → Huber Loss (共振峰加权)
```

## 模型架构

### 编码器

| 编码器 | 说明 | 适用场景 |
|--------|------|----------|
| `GNNEncoder` | **推荐** — GraphSAGE/GCN/GAT 图卷积 | 有网格拓扑 |
| `PointNetEncoder` | MLP + 全局池化 + 拼接 | 纯点云 |
| `DeepONetEncoder` | Branch/Trunk解耦 + SIREN空间编码 | 几何算子学习 |
| `SimplePointEncoder` | 逐点MLP baseline | 快速测试 |

### 解码器

`PerPointFRFDecoder`: 逐点特征 ⊕ sin/cos频率编码(64d) → MLP → 每频率标量  
`out_dim=1`: 位移FRF (有符号); `out_dim=2`: 复数 FRF (实部+虚部)

### 损失函数

- **Huber Loss (SmoothL1)**: 小误差平方收敛, 大误差线性防梯度爆炸
- **共振峰加权**: `weight = 1 + α·|target|`, α=2.0
- **频率平衡**: 按目标幅值反比加权, β=0.8 (小峰获得更高权重)
- **复数约束**: `loss = huber(Re) + huber(Im) + 2.0·huber(Amplitude)`

### 数据归一化

- FRF: **asinh 对数压缩** — 保留共振峰相对优势, 防梯度爆炸
- 频率: 线性映射到 [-1, 1]
- 逆变换: `torch.sinh(frf)` 或 `dataset.undo_normalize(frf)`

## 快速开始

```bash
cd f:\毕业论文\q\Learning_Vibrating_Plates-main

# 1. 生成合成数据 (3D悬臂梁, 240节点, 120频率, 2个共振峰, 有符号位移FRF)
F:\pytorch_cuda12\python.exe geometric_frf\sample\generate_data.py

# 2. 查看真实FRF长什么样
F:\pytorch_cuda12\python.exe geometric_frf\sample\测试.py

# 3. 训练
F:\pytorch_cuda12\python.exe geometric_frf\sample\run_validation.py

# 4. 评估 + 可视化
F:\pytorch_cuda12\python.exe geometric_frf\sample\evaluate.py
```

## HDF5 数据格式

| 键名 | 形状 | 必需 | 说明 |
|------|------|------|------|
| `points` | (S, N, 3) | ✓ | 三维节点坐标 |
| `point_frf` | (S, N, F) | ✓ | 逐点FRF (原始物理值) |
| `frequencies` | (S, F) | ✓ | 频率值 (Hz) |
| `edges` | (S, 2, E) | 可选 | 网格拓扑 (GNN用, int64) |
| `point_features` | (S, N, D) | 可选 | 逐点材料特征 |
| `phy_para` | (S, G) | 可选 | 全局物理参数 (自动广播) |

## 使用训练好的模型

```python
import numpy as np
data = np.load('output/final_results.npz')

# 第s个样本, 第i个三维节点
x, y, z   = data['points'][s, i]           # 坐标
frf_pred  = data['predicted_frf'][s, i]    # 预测FRF
frf_true  = data['target_frf'][s, i]       # 真实FRF
freq_axis = data['frequencies']            # 频率轴 (Hz)
```

## 依赖

- PyTorch + PyTorch Geometric
- NumPy, h5py, PyYAML, Matplotlib

## 当前训练结果

| 指标 | 结果 |
|------|------|
| 数据 | 3D 悬臂梁, 240 节点, 120 频率, 200/50/50 划分, 有符号位移FRF |
| 模型 | GNN GraphSAGE, 1.07M 参数, LeakyReLU激活 |
| 共振频率定位 | 待验证 (两峰: ~65 Hz / ~409 Hz) |
| 损失函数 | Huber + 共振峰加权(α=2.0) + 频率平衡(β=0.8) |
| 修复项 | 自适应峰值采样, ReLU→LeakyReLU, freq_balance→0.8 |
| 固定端约束 | 去掉几何噪声 + 边界损失 (权重50.0) |
