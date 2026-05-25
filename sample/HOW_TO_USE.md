# geometric_frf 使用说明

## 1. 目录结构

```
geometric_frf/
├── models/
│   ├── modal_model.py        DeepONet-SIREN 模态参数预测模型
│   ├── geometric_encoder.py  编码器 (GNN/PointNet/DeepONet)
│   ├── perpoint_decoder.py   逐点FRF解码器 (GNN等旧方案用)
│   ├── frf_model.py          模型工厂 build_geometric_model()
│   ├── geometry_data.py      GeometryData 数据容器
│   └── siren.py              SIREN 正弦激活
├── data/
│   └── dataset.py            HDF5数据集 (asinh归一化) + collate
├── training/
│   ├── losses.py             modal_loss — 模态参数MSE
│   └── trainer.py            训练循环 + 评估
├── configs/
│   └── modal_frf.yaml        当前配置
├── sample/
│   ├── data/                 合成HDF5数据
│   ├── generate_data.py      数据生成 (Euler-Bernoulli物理模型)
│   ├── run_validation.py     训练脚本
│   ├── evaluate.py           评估 + 保存结果
│   ├── 测试.py               查看真实FRF
│   ├── 对比图.py             预测vs真实对比
│   ├── predict.py            推理
│   └── output/               (checkpoint + 图表)
```

## 2. 数据

### 生成机制

`generate_data.py` 使用 Euler-Bernoulli 梁理论：
- 随机化 E, ρ, L, b, h → 计算固有频率 ω_k, 阻尼比 ζ_k, 模态振型 φ_k(x)
- 自适应频率网格: 每个共振峰 ±3·半功率带宽内密集采样
- 输出: point_frf (240,120,2) + modal_omega (2,) + modal_zeta (2,) + modal_phi (240,2)

### HDF5 格式

| 键 | 形状 | 说明 |
|------|------|------|
| points | (S, N, 3) | 节点坐标 |
| point_frf | (S, N, F, 2) | 复数FRF [Re, Im] |
| frequencies | (S, F) | 频率 Hz |
| point_features | (S, N, 11) | 3局部+8全局特征 |
| edges | (S, 2, E) | 网格拓扑 |
| modal_omega | (S, 2) | 固有圆频率 rad/s |
| modal_zeta | (S, 2) | 阻尼比 |
| modal_phi | (S, N, 2) | 模态振型 |

### 归一化

- FRF: asinh 对数压缩 → 评估时 torch.sinh() 还原
- 频率: 线性映射到 [-1, 1]

## 3. 模型

### DeepONet-SIREN 模态参数预测

```
Trunk (SIREN):  (x,y,z) → SirenMLP(w0=30) → [φ_1, φ_2]  模态振型(N,2)
Branch (MLP):   point_features → MLP → pool → [ω_1,ζ_1, ω_2,ζ_2]  (4,)
PhysicsDecoder: φ + ω + ζ + freq → FRF(Re,Im) (无参数)
```

损失: `MSE(ω) + MSE(ζ) + MSE(φ)`

## 4. 使用

```bash
cd F:\毕业论文\geometric_frf

# 生成数据
F:\pytorch_cuda12\python.exe sample\generate_data.py

# 查看数据
F:\pytorch_cuda12\python.exe sample\测试.py

# 训练
F:\pytorch_cuda12\python.exe sample\run_validation.py

# 评估
F:\pytorch_cuda12\python.exe sample\evaluate.py

# 对比图
F:\pytorch_cuda12\python.exe sample\对比图.py
```

## 5. 当前配置

| 参数 | 值 |
|------|-----|
| 模型 | DeepONet-SIREN, 256隐层, 4层Trunk |
| 参数 | 334,598 |
| 损失 | modal_loss (ω÷1000 + ζ×10⁵ + φ×10) |
| 学习率 | 0.0005, CosineAnnealing |
| 训练 | 32×2000 epochs |
| 数据 | 200训练/50验证/50测试, 几何±10%随机 |
