"""
dataset.py — 几何数据 Dataset + DataLoader。

支持的 HDF5 键:
    'points':         节点坐标 (n_samples, n_points, 3)
    'point_features': 节点特征 (n_samples, n_points, F) [可选]
    'edges':          单元拓扑 (n_samples, 2, n_edges) [可选]
    'point_frf':      逐点FRF目标 (n_samples, n_points, n_freqs)
    'frequencies':    频率值 (n_samples, n_freqs)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os

from ..models.geometry_data import GeometryData


class GeometricHDF5Dataset(Dataset):
    """
    几何数据 HDF5 数据集。

    输入:  点坐标 + 可选几何特征
    输出:  逐点 FRF (每个空间点一条频响曲线)
    """

    def __init__(self, data_paths, config, data_dir=".",
                 test=False, normalization=True,
                 sample_idx=slice(None), freq_idx=None):
        """
        参数:
            data_paths:    HDF5 文件路径列表 (相对于 data_dir)
            config:        配置对象 (dict)
            data_dir:      数据根目录
            test:          测试模式 (关闭频率采样)
            normalization: 是否归一化
            sample_idx:    样本索引切片
            freq_idx:      频率采样索引
        """
        self.data_paths = [os.path.join(data_dir, p) for p in data_paths]
        self.config = config
        self.normalization = normalization
        self.test = test
        self.freq_sampling = (not test) and config.get('freq_sample', False)
        self.freq_sampling_limit = config.get('freq_limit', 300)

        self.required_keys = {'points', 'point_frf', 'frequencies'}
        self.optional_keys = {'point_features', 'edges', 'phy_para'}
        self.all_keys = self.required_keys | self.optional_keys

        # 加载所有数据到内存
        loaded = {}
        for path in self.data_paths:
            with h5py.File(path, 'r') as f:
                for key in self.all_keys:
                    try:
                        raw = f[key][:]
                        # edges 保持 int64, 其余转为 float32
                        if key == 'edges':
                            data = torch.from_numpy(raw).long()
                        else:
                            data = torch.from_numpy(raw).float()
                        if key in loaded:
                            loaded[key] = torch.cat([loaded[key], data], dim=0)
                        else:
                            loaded[key] = data
                    except (KeyError, ValueError):
                        pass

        # 应用样本索引
        for key in loaded:
            if freq_idx is not None and key in {'point_frf', 'frequencies'}:
                loaded[key] = loaded[key][sample_idx][:, :, freq_idx] if loaded[key].ndim == 3 \
                    else loaded[key][sample_idx][:, freq_idx]
            else:
                loaded[key] = loaded[key][sample_idx]

        self.loaded = loaded

        if self.normalization:
            self._normalize_frequencies()
            self._normalize_point_frf()

    def _normalize_frequencies(self):
        """
        频率归一化到 [-1, 1]。

        频率是已知物理量 (1-500 Hz)，用统一物理极值映射，不依赖数据 min/max。
        保证训练/验证/测试集使用完全相同的映射公式。
        """
        if 'frequencies' in self.loaded:
            f_min, f_max = 1.0, 500.0   # 物理频率范围 (Hz)
            self.loaded['frequencies'] = (self.loaded['frequencies'] - f_min) / (f_max - f_min) * 2 - 1

    def _normalize_point_frf(self):
        """
        asinh 对数压缩归一化。

        FRF 动态范围极大 (共振峰可达非共振区的数百倍)。
        z-score 会使共振峰变为 200+ 的离群点，导致梯度爆炸。
        asinh 在原点附近近似线性，在大值区域近似对数，平滑压缩量级。

        逆变换: torch.sinh(data)
        """
        if 'point_frf' in self.loaded:
            data = self.loaded['point_frf']  # (S, N, F)
            self.loaded['point_frf'] = torch.asinh(data)

    def undo_normalize(self, frf):
        """还原 FRF 到原始尺度 (asinh 逆变换)"""
        return torch.sinh(frf)

    def __len__(self):
        return len(self.loaded.get('point_frf', self.loaded['points']))

    def __getitem__(self, idx):
        points = self.loaded['points'][idx]

        # 逐点特征拼接: point_features + 广播的全局 phy_para
        point_features = None
        feat_parts = []
        if 'point_features' in self.loaded:
            feat_parts.append(self.loaded['point_features'][idx])
        if 'phy_para' in self.loaded:
            phy = self.loaded['phy_para'][idx]  # (D_global,)
            feat_parts.append(phy.unsqueeze(0).expand(points.shape[0], -1))
        if feat_parts:
            point_features = torch.cat(feat_parts, dim=-1)  # (N, D_local + D_global)

        # 可选边拓扑
        edge_index = None
        if 'edges' in self.loaded:
            edge_index = self.loaded['edges'][idx]

        geometry = GeometryData(
            points=points,
            point_features=point_features,
            edge_index=edge_index,
            batch=None,
        )

        freq = self.loaded['frequencies'][idx]
        target = self.loaded['point_frf'][idx]

        if self.freq_sampling:
            # 自适应频率网格已天然在峰值区域密集采样, 此处均匀随机即可
            n_total = len(freq)
            n_sample = min(self.freq_sampling_limit, n_total)
            freq_idx = torch.randperm(n_total)[:n_sample]
            freq = freq[freq_idx]
            target = target[:, freq_idx]

        return {'geometry': geometry, 'point_frf': target, 'frequencies': freq}


def collate_geometry_batch(batch):
    """
    批次整理函数。

    相同点数 → stack 为 (B, N, *)   |   不同点数 → 拼接 + batch 索引

    edge_index 批处理 (GNN 兼容):
        对每个样本的边索引加上节点偏移量，使得不同样本的图互不连接。
        例如: 样本0的节点 [0, N0), 样本1的节点 [N0, N0+N1), ...
        样本i的边索引上所有值 += cumulative_offset[i]
    """
    n_points_list = [item['geometry'].points.shape[0] for item in batch]
    all_same_n = all(n == n_points_list[0] for n in n_points_list)

    if all_same_n:
        N = n_points_list[0]
        B = len(batch)
        points = torch.stack([item['geometry'].points for item in batch])
        point_frf = torch.stack([item['point_frf'] for item in batch])
        frequencies = torch.stack([item['frequencies'] for item in batch])

        point_feat = None
        if batch[0]['geometry'].point_features is not None:
            point_feat = torch.stack([item['geometry'].point_features for item in batch])

        # —— 批处理 edge_index (GNN兼容) ——
        # 将每个样本的边索引加节点偏移后拼接
        edge_idx = None
        batch_tensor = None
        if batch[0]['geometry'].edge_index is not None:
            all_edges = []
            for i, item in enumerate(batch):
                ei = item['geometry'].edge_index.clone()  # (2, E_i)
                ei = ei + i * N  # 节点偏移
                all_edges.append(ei)
            edge_idx = torch.cat(all_edges, dim=1)  # (2, B*E)
            # 生成 batch 索引供 GNN 全局池化
            batch_tensor = torch.arange(B).repeat_interleave(N)

        geometry = GeometryData(points=points, point_features=point_feat,
                                edge_index=edge_idx, batch=batch_tensor)
    else:
        all_points, all_features, all_frfs, all_batch = [], [], [], []
        all_edges = []
        frequencies = torch.stack([item['frequencies'] for item in batch])
        cumulative_offset = 0

        for i, item in enumerate(batch):
            n_pts = item['geometry'].points.shape[0]
            all_points.append(item['geometry'].points)
            all_frfs.append(item['point_frf'])
            all_batch.append(torch.full((n_pts,), i, dtype=torch.long))
            if item['geometry'].point_features is not None:
                all_features.append(item['geometry'].point_features)

            # 边索引 + 节点偏移 (GNN兼容)
            if item['geometry'].edge_index is not None:
                ei = item['geometry'].edge_index.clone()
                ei = ei + cumulative_offset
                all_edges.append(ei)

            cumulative_offset += n_pts

        points = torch.cat(all_points, dim=0)
        point_frf = torch.cat(all_frfs, dim=0)
        point_feat = torch.cat(all_features, dim=0) if all_features else None
        batch_tensor = torch.cat(all_batch, dim=0)
        edge_idx = torch.cat(all_edges, dim=1) if all_edges else None

        geometry = GeometryData(points=points, point_features=point_feat,
                                edge_index=edge_idx, batch=batch_tensor)

    return {'geometry': geometry, 'point_frf': point_frf, 'frequencies': frequencies}


def get_geometric_dataloader(args, config, data_dir=".", num_workers=0,
                             shuffle=True, normalization=True):
    """
    构建训练/验证/测试 DataLoader。

    返回: trainloader, valloader, testloader, trainset, valset, testset
    """
    batch_size = args.batch_size
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 随机划分
    idx = torch.randperm(config.get('n_samples', 5000))
    train_idx = idx[:config.get('n_train_samples', 4000)].numpy()
    val_idx = idx[-config.get('n_val_samples', 500):].numpy()

    trainset = GeometricHDF5Dataset(
        config['data_path_train'], config, data_dir=data_dir,
        normalization=normalization, test=False, sample_idx=train_idx,
    )
    valset = GeometricHDF5Dataset(
        config['data_path_val'], config, data_dir=data_dir,
        normalization=normalization, test=True,
        sample_idx=val_idx if config.get('n_val_samples', 0) > 0 else slice(None),
    )
    testset = None
    if config.get('data_paths_test') is not None:
        testset = GeometricHDF5Dataset(
            config['data_paths_test'], config, data_dir=data_dir,
            normalization=normalization, test=True,
        )
    else:
        testset = valset

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, drop_last=shuffle, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_geometry_batch,
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=2, drop_last=False, shuffle=False,
        num_workers=num_workers, collate_fn=collate_geometry_batch,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=2, drop_last=False, shuffle=False,
        num_workers=num_workers, collate_fn=collate_geometry_batch,
    )

    return trainloader, valloader, testloader, trainset, valset, testset
