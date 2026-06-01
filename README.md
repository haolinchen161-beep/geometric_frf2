# geometric_frf — 基于几何的频响函数预测

输入 3D 几何 → DeepONet-SIREN → 模态参数 (ω, ζ, φ) → 物理公式重建 FRF。

## 1. 目录结构

```
├── models/
│   ├── modal_model.py        DeepONet-SIREN (PhysicsDecoder + ModalFRFModel)
│   ├── frf_model.py          模型工厂 build_geometric_model()
│   ├── geometry_data.py      GeometryData 数据容器
│   └── siren.py              SIREN 正弦激活 (w0=30)
├── data/
│   └── dataset.py            HDF5 数据集 (flat + per-sample-group) + collate
├── training/
│   ├── losses.py             modal_loss + frf_loss
│   └── trainer.py            三阶段训练循环 + 评估 (支持可变N/F)
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

## 2. 整体流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                        数据准备阶段                                  │
│  ANSYS MAPDL                                                        │
│  ├─ Sobol 采样: E, ρ, L, W, H (±5%~10% 随机化)                      │
│  ├─ 建模 → SOLID187 网格 → 模态分析 (LANB)                          │
│  ├─ 提取: ω_k, ζ_k, φ_k(x), φ_k(x_f)                              │
│  ├─ 物理公式计算 FRF: H(x,x_f,ω) = Σ φ_k(x)·φ_k(x_f)/(ω_k²-ω²+j2ζ_kω_kω) │
│  └─ 输出: train/val/test.h5 (per-sample-group 格式, ~4k节点/样本)    │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        数据加载阶段                                  │
│  GeometricHDF5Dataset                                               │
│  ├─ 检测格式: group (per-sample-group) 或 flat                      │
│  ├─ 归一化: FRF → asinh(), 频率 → [-1, 1]                          │
│  └─ collate_geometry_batch: 同节点数→stack, 不同→拼接               │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        模型前向传播                                  │
│                                                                     │
│  输入:                                                              │
│    geometry_data = {points(N,3), point_features(N,6), batch(N,)}    │
│    frequencies = (B, F)  归一化频率 [-1, 1]                         │
│    phi_exc = (B, K)      激励点振型值                               │
│                                                                     │
│  ┌─────────────────────┐    ┌──────────────────────────────┐       │
│  │  Trunk (SIREN 编码器) │    │  Branch (MLP 编码器)          │       │
│  │                      │    │                              │       │
│  │  (x,y,z) → 归一化    │    │  point_features(N,6)         │       │
│  │    ↓                 │    │    + physics_prior(N,1)      │       │
│  │  SirenMLP            │    │    = br_in(N,7)              │       │
│  │  4层 × sin(w0·x)     │    │    ↓                        │       │
│  │  3→256→256→256→256   │    │  Linear(7→64) + LeakyReLU   │       │
│  │    ↓                 │    │  Linear(64→64)               │       │
│  │  spatial_feat(N,256) │    │    ↓                        │       │
│  │    ↓                 │    │  global_mean_pool → (B,64)   │       │
│  │  head_phi(256→2)     │    │    ↓                        │       │
│  │    ↓                 │    │  head_modal(64→4)            │       │
│  │  φ_k(x) (N, K=2)    │    │    ↓                        │       │
│  └─────────────────────┘    │  Δω, ζ (B, K×2)             │       │
│                              └──────────────────────────────┘       │
│                                         ↓                           │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  ω 跳跃连接 (物理先验 + MLP 微调)                         │       │
│  │                                                          │       │
│  │  physics = (H/L²)·√(E/ρ)    ← 材料/几何先验              │       │
│  │    ↓                                                     │       │
│  │  skip_omega(physics) → softplus × 20000  = ω_coarse      │       │
│  │  tanh(Δω) × 8000                           = ω_fine      │       │
│  │  ω = ω_coarse + ω_fine                                   │       │
│  │  ζ = softplus(ζ_raw) × 0.004 + 1e-4                     │       │
│  └──────────────────────────────────────────────────────────┘       │
│                              ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  PhysicsDecoder (无参数物理解码器)                         │       │
│  │                                                          │       │
│  │  输入: φ(N,K), ω(B,K), ζ(B,K), freq(B,F), φ_exc(B,K)   │       │
│  │                                                          │       │
│  │  H_k = φ_k(x)·φ_k(x_f) / (ω_k² - ω² + j·2ζ_k·ω_k·ω)  │       │
│  │  FRF = amp_scale × Σ_k H_k                               │       │
│  │                                                          │       │
│  │  输出: (total_N, F, 2)  [Re, Im]                         │       │
│  └──────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        三阶段训练                                    │
│                                                                     │
│  阶段1 (epoch 0~499):   独立回归                                    │
│    loss = rel_MSE(ω)×500K + rel_MSE(ζ)×1K + MSE(φ)×1K             │
│    → 学习模态参数的基本分布                                          │
│                                                                     │
│  阶段2 (epoch 500~1499): 冻φ攻ω (冻结 trunk + head_phi)            │
│    loss = 上述 + frf_loss×20                                        │
│    → 通过 FRF 物理约束对齐 ω, ζ                                    │
│                                                                     │
│  阶段3 (epoch 1500+):   解冻联调                                    │
│    loss = 上述                                                      │
│    → 精修峰值高度, 端到端微调                                        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        评估与可视化                                  │
│  evaluate.py → 预测 vs 真实: asinh-MSE, 幅值MAE/MAPE, ω_MAE        │
│  对比图.py   → 3个点 × (幅值+实部+虚部) 对比图                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 3. 数据

### ANSYS 3D 悬臂板

| 参数 | 值 |
|------|-----|
| 板尺寸 | 100×60×10mm (铝, E=71.7GPa, ρ=2810) |
| 固定方式 | 固支端 x∈[0, 8mm] 全宽夹紧 |
| 激励点 | 自由端最远角 (L, W, H) |
| 模态 | 前 2 阶, 质量归一化振型 (ΦᵀMΦ=I) |
| 频率网格 | 40 点, 自适应 (共振峰 ±3·半功率带宽密集) |
| 样本 | 300 (200/50/50), Sobol 采样几何±10% 随机化 |
| 网格 | 6mm 自由四面体, ~4k 节点/样本 |

### HDF5 格式 (per-sample-group)

每个样本的节点数 N_nodes 和频率点数 F 可变, 不同样本可能不同。

```
/sample_0/
├── points         (N₀, 3)       节点三维坐标 [x, y, z] (m)
├── point_frf      (N₀, F₀, 2)  各节点的复数频响函数 [Re, Im]
├── frequencies    (F₀,)         频率采样点 (Hz), 自适应网格
├── point_features (6,)          全局几何/材料特征, 广播至每个节点:
│                                [E/E₀, ρ/ρ₀, L/L₀, W/W₀, H/H₀, n_modes]
├── modal_omega    (K,)          各阶固有圆频率 ω_k (rad/s), K=2
├── modal_zeta     (K,)          各阶阻尼比 ζ_k (=0.003)
├── modal_phi      (N₀, K)       各节点的模态振型 φ_k(x), 质量归一化
└── modal_phi_exc  (K,)          激励点处的振型值 φ_k(x_f)

/sample_1/
├── points         (N₁, 3)       ← N₁ 可能 ≠ N₀
├── point_frf      (N₁, F₁, 2)  ← F₁ 可能 ≠ F₀
└── ...
```

其中:
- N_nodes: 每个样本的有限元节点数, 因网格尺寸和几何尺寸变化而不同 (~3k~5k)
- F: 频率采样点数, 因自适应网格策略而不同 (~35~45)
- K=2: 模态阶数 (固定)

### FRF 公式

```
H(x, x_f, ω) = Σ_k φ_k(x)·φ_k(x_f) / (ω_k² - ω² + j·2ζ_k·ω_k·ω)
```

### 归一化

- FRF: `torch.asinh()` → 评估时 `torch.sinh()` 还原
- 频率: `(f - f_min) / (f_max - f_min) × 2 - 1` → 映射到 [-1, 1]

## 4. 模型详细结构

### 编码器: ModalFRFModel (DeepONet 架构)

| 组件 | 结构 | 输入 | 输出 |
|------|------|------|------|
| **Trunk (SIREN)** | 4层 SirenMLP, sin(w0=30·x) 激活 | 归一化坐标 (N, 3) | 空间特征 (N, 256) |
| **head_phi** | Linear(256→2) | 空间特征 (N, 256) | 模态振型 φ_k(x) (N, 2) |
| **Branch (MLP)** | Linear(7→64) + LeakyReLU + Linear(64→64) | 点特征+物理先验 (N, 7) | 逐点特征 (N, 64) |
| **global_mean_pool** | 按 batch 索引求均值 | 逐点特征 (N, 64) | 全局特征 (B, 64) |
| **head_modal** | Linear(64→4) | 全局特征 (B, 64) | Δω, Δζ (B, 4) |
| **skip_omega** | Linear(1→2), 初始化为 0 | 物理先验 (B, 1) | ω 基线 (B, 2) |

### ω, ζ 预测 (粗-细解耦)

```
physics = (H / L²) · √(E / ρ)          ← 材料/几何先验
ω_coarse = softplus(skip_omega(physics)) × 20000   ← 宏观基线
ω_fine   = tanh(Δω) × 8000                          ← MLP 微调
ω = ω_coarse + ω_fine

ζ = softplus(Δζ) × 0.004 + 1e-4
```

### 解码器: PhysicsDecoder (无参数)

直接用物理公式从模态参数重建 FRF, 不含可学习参数:

```
反归一化频率: f_phys = (freq + 1) / 2 × (f_max - f_min) + f_min
角频率: ω_q = 2π × f_phys

对每个模态 k:
  dw = ω_k² - ω_q²
  γ  = 2ζ_k · ω_k · ω_q
  D  = dw² + γ²
  H_k = amp_scale × (dw - jγ) / D

FRF = Σ_k φ_k(x) · φ_k(x_f) · H_k
```

## 5. 快速开始

```bash
# 生成数据 (需 ANSYS MAPDL license, ~数小时)
python ansys/generate_3d_test.py

# 查看原始 FRF
python sample/测试.py

# 训练
python sample/run_validation.py

# 评估
python sample/evaluate.py

# 对比图
python sample/对比图.py
```

## 6. 当前配置

| 参数 | 值 |
|------|-----|
| 模型 | DeepONet-SIREN |
| hidden_dim / branch_hidden_dim | 256 / 64 |
| trunk_layers / branch_layers | 4 / 2 |
| n_modes | 2 |
| siren_w0 | 30 |
| amp_scale | 500000 |
| freq_range | 1 ~ 5000 Hz |
| 损失 | rel_ω×500K + rel_ζ×1K + φ×1K + FRF×20 (阶段2+) |
| 优化器 | AdamW, lr=0.0005, weight_decay=5e-5, CosineAnnealing |
| gradient_clip | 2.0 (仅 Branch 侧) |
| batch_size | 8 |
| FP16 | 关闭 (SIREN+AMP 易 NaN) |
| 训练 | 200 样本 × 2000 epochs, 三阶段 |
