# Geometric FRF — 基于几何输入的逐点频响函数预测

将神经网络输入从二维图像改为**几何属性**（点坐标、网格拓扑、FEM网格等），
输出为**每个点的完整频响函数 (FRF)**。

## 目录结构

```
geometric_frf/
├── models/                     # 模型模块
│   ├── film.py                 #   FiLM 条件调制层
│   ├── geometry_data.py        #   统一几何数据容器 (GeometryData)
│   ├── geometric_encoder.py    #   几何编码器 (PointNet/Simple/GNN)
│   ├── perpoint_decoder.py     #   逐点FRF解码器 (频率查询 + 分块MLP)
│   └── frf_model.py            #   完整模型 + 便捷构建函数
├── data/                       # 数据模块
│   └── dataset.py              #   HDF5数据集 + collate + DataLoader
├── training/                   # 训练模块
│   └── trainer.py              #   训练循环 + 评估函数
├── configs/                    # 配置文件
│   ├── dataset.yaml            #   数据集训练配置
│   ├── pointnet_frf.yaml       #   PointNet 模型配置
│   ├── gnn_frf.yaml            #   GNN 模型配置 (预留)
│   └── simple_frf.yaml         #   Simple MLP baseline 配置
├── scripts/
│   └── run.py                  #   训练入口脚本
└── README.md
```

## 数据流

```
HDF5文件 → GeometricHDF5Dataset → GeometryData (点坐标+特征+拓扑)
         → GeometricEncoder → 逐点隐特征 (B, N, D)
         → PerPointFRFDecoder + 频率查询 → 逐点FRF (B, N, n_freqs)
         → MSE Loss
```

## 模型架构

### 编码器 (3种可选)

| 编码器 | 说明 |
|--------|------|
| `SimplePointEncoder` | 最简逐点MLP，baseline |
| `PointNetEncoder` | MLP + 全局池化 + 拼接，含全局上下文 |
| `GNNEncoder` | 图神经网络 (预留，需网格拓扑) |

### 解码器

`PerPointFRFDecoder`: 频率查询(FiLM) + 分块MLP → 每个点的完整FRF曲线

## 用法

```bash
F:\pytorch_cuda12\python.exe geometric_frf/scripts/run.py \
    --config configs/dataset.yaml \
    --model_cfg configs/pointnet_frf.yaml \
    --data_dir ./data \
    --batch_size 64
```

## HDF5 数据格式

```python
# 必需键
'points':       (n_samples, n_points, 3)     # 节点坐标
'point_frf':    (n_samples, n_points, n_freqs) # 逐点FRF目标
'frequencies':  (n_samples, n_freqs)          # 频率值 [1, 300]

# 可选键
'point_features': (n_samples, n_points, F)    # 节点特征
'edges':          (n_samples, 2, n_edges)     # 单元拓扑
'phy_para':       (n_samples, n_params)       # 全局物理参数
```

## 依赖

- PyTorch
- NumPy
- h5py
- PyYAML
