# geometric_frf 使用说明

## 1. 目录结构

```
geometric_frf/
├── models/
│   ├── film.py               FiLM 条件调制层
│   ├── geometry_data.py      GeometryData 数据容器
│   ├── geometric_encoder.py  编码器 (PointNet/GNN/DeepONet)
│   ├── perpoint_decoder.py   逐点FRF解码器 (Concat+sin编码)
│   ├── frf_model.py          完整模型
│   └── siren.py              SIREN 正弦激活
├── data/
│   └── dataset.py            HDF5数据集 (asinh归一化) + collate
├── training/
│   ├── losses.py             Huber Loss + 共振峰加权
│   └── trainer.py            训练循环 + 评估
├── configs/
│   ├── dataset.yaml
│   ├── pointnet_frf.yaml / gnn_frf.yaml / deeponet_frf.yaml
├── scripts/
│   └── run.py                标准训练入口
└── sample/
    ├── data/                 合成HDF5数据 (3D悬臂梁)
    ├── generate_data.py      数据生成 (Euler-Bernoulli物理模型)
    ├── run_validation.py     训练脚本
    ├── evaluate.py           训练后评估+可视化
    ├── 测试.py               查看真实FRF数据
    ├── predict.py            预测/推理
    └── output/               输出 (checkpoint + 图表 + 结果)
```

## 2. 数据准备

### 2.1 HDF5 格式

| 键名 | 形状 | 必需 | 说明 |
|------|------|------|------|
| `points` | (S, N, 3) | ✓ | 三维节点坐标 (x,y,z) |
| `point_frf` | (S, N, F) | ✓ | 逐点频响函数(原始物理值) |
| `frequencies` | (S, F) | ✓ | 频率值 (Hz) |
| `edges` | (S, 2, E) | 可选 | 网格拓扑 (GNN用) |
| `point_features` | (S, N, D) | 可选 | 逐点材料特征 |
| `phy_para` | (S, G) | 可选 | 全局物理参数 (广播到每点) |

### 2.2 数据归一化

`GeometricHDF5Dataset` 使用 **asinh 对数压缩** 而非 z-score:
- `target = asinh(raw_frf)` — 保留共振峰相对优势
- `freq = (freq - 1) / (500 - 1) * 2 - 1` — 线性映射到 [-1, 1]
- 逆变换: `raw = torch.sinh(target)` 或 `dataset.undo_normalize(target)`

### 2.3 示例数据

`generate_data.py` 生成 3D 悬臂梁 (Euler-Bernoulli 物理模型):
- 网格: 20×4×3 = 240 节点, 梁尺寸随机 ±10% (L、b、h 独立变化)
- 120 频率点, 自适应峰值采样 (每个样本在自身共振频率 ±3·半功率带宽内密集线性)
- 每样本 2 个固有频率 (随几何/材料变化, ~49~85 Hz / ~310~530 Hz), 非共振区平滑
- 有符号位移 FRF (保留相位信息, 支持颤振/模态分析)
- 铝材: E=69GPa±5%, ρ=2700kg/m³±3%, 样本间随机变化

## 3. 模型

### 3.1 编码器

| 类型 | 说明 |
|------|------|
| `gnn` | **推荐** — GraphSAGE 图卷积, 利用网格拓扑 |
| `pointnet` | PointNet 风格, 纯点云 |
| `simple` | 逐点MLP baseline |
| `deeponet` | Branch/Trunk 解耦 + SIREN 空间编码 |

### 3.2 解码器 (Concat 模式)

- 逐点特征 (256d) + sin/cos 频率编码 (64d) → MLP → FRF 值
- `out_dim=1`: 位移FRF (有符号); `out_dim=2`: 复数 FRF

### 3.3 损失函数

**Huber Loss** (SmoothL1): 小误差平方收敛, 大误差线性防梯度爆炸。
- `resonance_weight=2.0`: 共振峰自适应加权 (降低第一峰主导)
- `freq_balance_strength=0.8`: 频率平衡 (小峰获得更高权重)

## 4. 训练

```bash
cd f:\毕业论文\q\Learning_Vibrating_Plates-main

# 生成数据
F:\pytorch_cuda12\python.exe geometric_frf\sample\generate_data.py

# 查看真实FRF
F:\pytorch_cuda12\python.exe geometric_frf\sample\测试.py

# 训练 (2000 epochs)
F:\pytorch_cuda12\python.exe geometric_frf\sample\run_validation.py
```

## 5. 评估 + 可视化

```bash
# 完整评估 (加载checkpoint → 预测 → 逆变换 → 图表)
F:\pytorch_cuda12\python.exe geometric_frf\sample\evaluate.py
```

输出:
- `output/evaluation.png` — 多点FRF对比 + 散点图 + 误差分布
- `output/final_compare.png` — 预测 vs 真实 (5位置 + 同x不同yz验证)
- `output/final_results.npz` — 全部测试样本的逐点物理FRF

## 6. 使用结果

```python
import numpy as np
data = np.load('output/final_results.npz')

# 第s个样本, 第i个三维节点
x, y, z   = data['points'][s, i]           # 坐标
frf_pred  = data['predicted_frf'][s, i]    # 预测的 FRF
frf_true  = data['target_frf'][s, i]       # 真实的 FRF
freq_axis = data['frequencies']            # 频率轴 (Hz)

# 画该点的FRF
import matplotlib.pyplot as plt
plt.semilogx(freq_axis, frf_pred, label='Predicted')
plt.semilogx(freq_axis, frf_true, label='Target')
plt.xlabel('Hz'); plt.ylabel('FRF'); plt.legend()
```

## 7. 当前配置

| 参数 | 值 |
|------|-----|
| 编码器 | GNN GraphSAGE, 4层, hidden=256, LeakyReLU (保留负值) |
| 解码器 | Concat+sin编码(64d), 4层, hidden=256 |
| 损失 | Huber + 共振峰加权(α=2.0) + 频率平衡(β=0.8) |
| 学习率 | 0.0005, CosineAnnealing |
| 批次 | 32, 2000 epochs |
| 归一化 | asinh 对数压缩 (天然支持正负值) |
| 数据 | 3D 悬臂梁, 240节点, 120频率, 200/50/50划分, 有符号位移FRF |
