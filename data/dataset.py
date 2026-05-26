"""
dataset.py — 几何数据 Dataset + DataLoader。

支持的 HDF5 格式:
  1. 扁平格式 (beam):  /points (S,N,3), /point_frf (S,N,F,2), ...
  2. 分组格式 (ANSYS): /sample_0/points, /sample_0/point_frf, ...
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os

from ..models.geometry_data import GeometryData


class GeometricHDF5Dataset(Dataset):
    """几何数据 HDF5 数据集, 自动检测扁平/分组格式."""

    def __init__(self, data_paths, config, data_dir=".",
                 test=False, normalization=True):
        self.config = config
        self.normalization = normalization
        self.test = test
        self.freq_min = config.get('freq_min', 1.0)
        self.freq_max = config.get('freq_max', 8000.0)
        self._format = None  # 'flat' or 'group'
        self._loaded = None   # flat: 预加载字典
        self._samples = None  # group: [(file_path, group_name), ...]

        full_paths = [os.path.join(data_dir, p) for p in data_paths]
        self._detect_format(full_paths)

    def _detect_format(self, full_paths):
        with h5py.File(full_paths[0], 'r') as f:
            keys = list(f.keys())
            if any(k.startswith('sample_') for k in keys):
                self._format = 'group'
            else:
                self._format = 'flat'

        if self._format == 'group':
            self._samples = []
            for fp in full_paths:
                with h5py.File(fp, 'r') as f:
                    for key in sorted(f.keys(), key=lambda k: int(k.split('_')[-1])):
                        if key.startswith('sample_'):
                            self._samples.append((fp, key))
        else:
            self._loaded = {}
            required_keys = {'points', 'point_frf', 'frequencies'}
            optional_keys = {'point_features', 'edges', 'phy_para',
                             'modal_omega', 'modal_zeta', 'modal_phi',
                             'modal_phi_exc'}
            for fp in full_paths:
                with h5py.File(fp, 'r') as f:
                    for key in required_keys | optional_keys:
                        try:
                            raw = f[key][:]
                            data = torch.from_numpy(raw).long() if key == 'edges' else torch.from_numpy(raw).float()
                            if key in self._loaded:
                                self._loaded[key] = torch.cat([self._loaded[key], data], dim=0)
                            else:
                                self._loaded[key] = data
                        except (KeyError, ValueError):
                            pass

            if self.normalization:
                if 'frequencies' in self._loaded:
                    self._loaded['frequencies'] = (self._loaded['frequencies'] - self.freq_min) / (self.freq_max - self.freq_min) * 2 - 1
                if 'point_frf' in self._loaded:
                    self._loaded['point_frf'] = torch.asinh(self._loaded['point_frf'])

    def undo_normalize(self, frf):
        return torch.sinh(frf)

    def __len__(self):
        if self._format == 'group':
            return len(self._samples)
        return len(self._loaded.get('point_frf', self._loaded['points']))

    def __getitem__(self, idx):
        if self._format == 'group':
            return self._get_group_item(idx)
        return self._get_flat_item(idx)

    def _get_group_item(self, idx):
        fp, grp_name = self._samples[idx]
        with h5py.File(fp, 'r') as f:
            grp = f[grp_name]
            points = torch.from_numpy(grp['points'][:]).float()
            freqs = torch.from_numpy(grp['frequencies'][:]).float()
            frf = torch.from_numpy(grp['point_frf'][:]).float()

            point_feat = None
            if 'point_features' in grp:
                gf = torch.from_numpy(grp['point_features'][:]).float()
                point_feat = gf.unsqueeze(0).expand(points.shape[0], -1)

            out = {'modal_omega': None, 'modal_zeta': None, 'modal_phi': None,
                   'modal_phi_exc': None}
            for key in ['modal_omega', 'modal_zeta', 'modal_phi', 'modal_phi_exc']:
                if key in grp:
                    out[key] = torch.from_numpy(grp[key][:]).float()

        if self.normalization:
            freqs = (freqs - self.freq_min) / (self.freq_max - self.freq_min) * 2 - 1
            frf = torch.asinh(frf)

        geometry = GeometryData(points=points, point_features=point_feat)
        result = {'geometry': geometry, 'point_frf': frf, 'frequencies': freqs}
        for key in ['modal_omega', 'modal_zeta', 'modal_phi', 'modal_phi_exc']:
            if out[key] is not None:
                result[key] = out[key]
        return result

    def _get_flat_item(self, idx):
        points = self._loaded['points'][idx]
        point_features = None
        feat_parts = []
        if 'point_features' in self._loaded:
            feat_parts.append(self._loaded['point_features'][idx])
        if 'phy_para' in self._loaded:
            phy = self._loaded['phy_para'][idx]
            feat_parts.append(phy.unsqueeze(0).expand(points.shape[0], -1))
        if feat_parts:
            point_features = torch.cat(feat_parts, dim=-1)

        edge_index = None
        if 'edges' in self._loaded:
            edge_index = self._loaded['edges'][idx]

        geometry = GeometryData(points=points, point_features=point_features,
                                edge_index=edge_index, batch=None)
        freq = self._loaded['frequencies'][idx]
        target = self._loaded['point_frf'][idx]

        out = {'geometry': geometry, 'point_frf': target, 'frequencies': freq}
        for key in ['modal_omega', 'modal_zeta', 'modal_phi', 'modal_phi_exc']:
            if key in self._loaded:
                out[key] = self._loaded[key][idx]
        return out


def collate_geometry_batch(batch):
    """批次整理: 同节点数→stack, 不同→拼接. 可变F→list."""
    n_points_list = [item['geometry'].points.shape[0] for item in batch]
    all_same_n = all(n == n_points_list[0] for n in n_points_list)
    f_lens = [item['frequencies'].shape[0] for item in batch]
    all_same_f = all(f == f_lens[0] for f in f_lens)

    # 频率/FRF: 同F则stack, 不同F则保留list
    if all_same_f:
        frequencies = torch.stack([item['frequencies'] for item in batch])
        if all_same_n:
            point_frf = torch.stack([item['point_frf'] for item in batch])
            points = torch.stack([item['geometry'].points for item in batch])
            point_feat = torch.stack([item['geometry'].point_features for item in batch]) \
                         if batch[0]['geometry'].point_features is not None else None
            edge_idx = None; batch_tensor = None
            if batch[0]['geometry'].edge_index is not None:
                all_edges = []
                for i, item in enumerate(batch):
                    ei = item['geometry'].edge_index.clone() + i * n_points_list[0]
                    all_edges.append(ei)
                edge_idx = torch.cat(all_edges, dim=1)
                batch_tensor = torch.arange(len(batch)).repeat_interleave(n_points_list[0])
            geometry = GeometryData(points=points, point_features=point_feat,
                                    edge_index=edge_idx, batch=batch_tensor)
        else:
            all_points, all_features, all_frfs, all_batch = [], [], [], []
            all_edges = []
            cumulative_offset = 0
            for i, item in enumerate(batch):
                n_pts = item['geometry'].points.shape[0]
                all_points.append(item['geometry'].points)
                all_frfs.append(item['point_frf'])
                all_batch.append(torch.full((n_pts,), i, dtype=torch.long))
                if item['geometry'].point_features is not None:
                    all_features.append(item['geometry'].point_features)
                if item['geometry'].edge_index is not None:
                    ei = item['geometry'].edge_index.clone() + cumulative_offset
                    all_edges.append(ei)
                cumulative_offset += n_pts
            points = torch.cat(all_points, dim=0)
            point_frf = torch.cat(all_frfs, dim=0)
            point_feat = torch.cat(all_features, dim=0) if all_features else None
            batch_tensor = torch.cat(all_batch, dim=0)
            edge_idx = torch.cat(all_edges, dim=1) if all_edges else None
            geometry = GeometryData(points=points, point_features=point_feat,
                                    edge_index=edge_idx, batch=batch_tensor)
    else:
        # 可变F: FRF和频率不可stack
        frequencies = [item['frequencies'] for item in batch]
        point_frf = [item['point_frf'] for item in batch]
        # geometry 走可变N拼接路径
        all_points, all_features, all_batch = [], [], []
        cumulative_offset = 0
        for i, item in enumerate(batch):
            n_pts = item['geometry'].points.shape[0]
            all_points.append(item['geometry'].points)
            all_batch.append(torch.full((n_pts,), i, dtype=torch.long))
            if item['geometry'].point_features is not None:
                all_features.append(item['geometry'].point_features)
            cumulative_offset += n_pts
        points = torch.cat(all_points, dim=0)
        point_feat = torch.cat(all_features, dim=0) if all_features else None
        batch_tensor = torch.cat(all_batch, dim=0)
        geometry = GeometryData(points=points, point_features=point_feat,
                                edge_index=None, batch=batch_tensor)

    out = {'geometry': geometry, 'point_frf': point_frf, 'frequencies': frequencies}
    modal = _stack_modal(batch)
    if modal:
        out.update(modal)
    return out


def _stack_modal(batch):
    for key in ['modal_omega', 'modal_zeta', 'modal_phi']:
        if key not in batch[0] or batch[0][key] is None:
            return {}
    # omega/ζ 可stack (同K), φ 需cat (N可变)
    result = {}
    for key in ['modal_omega', 'modal_zeta', 'modal_phi_exc']:
        if key in batch[0] and batch[0][key] is not None:
            result[key] = torch.stack([item[key] for item in batch])
    result['modal_omega'] = torch.stack([item['modal_omega'] for item in batch])
    result['modal_zeta'] = torch.stack([item['modal_zeta'] for item in batch])
    result['modal_phi'] = torch.cat([item['modal_phi'] for item in batch], dim=0)
    return result


def get_geometric_dataloader(args, config, data_dir=".", num_workers=0,
                             shuffle=True, normalization=True):
    batch_size = args.batch_size
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    idx = torch.randperm(config.get('n_samples', 5000))
    train_idx = idx[:config.get('n_train_samples', 4000)].numpy()
    val_idx = idx[-config.get('n_val_samples', 500):].numpy()

    trainset = GeometricHDF5Dataset(
        config['data_path_train'], config, data_dir=data_dir,
        normalization=normalization, test=False,
    )
    valset = GeometricHDF5Dataset(
        config['data_path_val'], config, data_dir=data_dir,
        normalization=normalization, test=True,
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
