"""
geometric_encoder.py — 几何编码器模块。

将几何数据编码为逐点隐特征 (B, N, D)。

提供:
    - GeometricEncoder:  抽象基类
    - SimplePointEncoder: 最简逐点MLP (baseline)
    - PointNetEncoder:    PointNet风格 (MLP + 全局池化 + 拼接)
    - GNNEncoder:         图神经网络编码器 (预留)
    - DeepONetEncoder:    Branch/Trunk解耦架构 (SIREN坐标编码 + 宏观特征)
"""

import torch
import torch.nn as nn
from .geometry_data import GeometryData
from .siren import SirenMLP


# ============================================================
# 抽象基类
# ============================================================
class GeometricEncoder(nn.Module):
    """几何编码器抽象基类。输入 GeometryData，输出逐点隐特征 (B, N, out_dim)。"""

    def __init__(self, in_channels: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

    def forward(self, geometry_data: GeometryData) -> torch.Tensor:
        raise NotImplementedError("子类必须实现 forward 方法")


# ============================================================
# 实现1: SimplePointEncoder — 最简逐点MLP (baseline)
# ============================================================
class SimplePointEncoder(GeometricEncoder):
    """
    最简几何编码器：逐点MLP映射，无全局特征聚合。
    适用场景: 快速baseline测试，点之间关联较弱的任务。
    """

    def __init__(self, in_channels: int = 0, hidden_dim: int = 256, out_dim: int = 256,
                 coord_dim: int = 3, n_layers: int = 4):
        super().__init__(in_channels, hidden_dim, out_dim)
        input_dim = coord_dim + in_channels

        layers = []
        current_dim = input_dim
        for i in range(n_layers):
            if i == n_layers - 1:
                layers.append(nn.Linear(current_dim, out_dim))
            else:
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.ReLU())
                current_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, geometry_data: GeometryData) -> torch.Tensor:
        points = geometry_data.points  # (B, N, 3)
        if geometry_data.point_features is not None:
            x = torch.cat([points, geometry_data.point_features], dim=-1)
        else:
            x = points
        return self.mlp(x)  # (B, N, out_dim)


# ============================================================
# 实现2: PointNetEncoder — MLP + 全局池化 + 拼接
# ============================================================
class PointNetEncoder(GeometricEncoder):
    """
    PointNet 风格几何编码器。
    逐点MLP提取局部特征 → 全局最大池化 → 与局部特征拼接 → 融合MLP。
    使每个点的表示同时包含局部几何和全局上下文信息。

    参考: Qi et al., "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"
    """

    def __init__(self, in_channels: int = 0, hidden_dim: int = 256, out_dim: int = 256,
                 coord_dim: int = 3, n_layers: int = 3, global_feat_dim: int = 256):
        super().__init__(in_channels, hidden_dim, out_dim)
        input_dim = coord_dim + in_channels

        # 逐点局部特征提取
        local_layers = []
        current_dim = input_dim
        for i in range(n_layers):
            local_layers.append(nn.Linear(current_dim, hidden_dim))
            local_layers.append(nn.ReLU())
            current_dim = hidden_dim
        self.local_mlp = nn.Sequential(*local_layers)

        # 全局特征提取 (max pooling + MLP)
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim, global_feat_dim),
            nn.ReLU(),
            nn.Linear(global_feat_dim, global_feat_dim),
        )

        # 局部+全局融合
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim + global_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, geometry_data: GeometryData) -> torch.Tensor:
        points = geometry_data.points  # (B, N, 3)
        if geometry_data.point_features is not None:
            x = torch.cat([points, geometry_data.point_features], dim=-1)
        else:
            x = points

        # 1. 逐点局部特征 (B, N, hidden_dim)
        local_features = self.local_mlp(x)

        # 2. 全局特征: 沿点维度最大池化 (B, global_feat_dim)
        global_features = local_features.max(dim=1)[0]
        global_features = self.global_mlp(global_features)

        # 3. 拼接+融合 (B, N, out_dim)
        global_expanded = global_features.unsqueeze(1).expand(-1, local_features.size(1), -1)
        fused = torch.cat([local_features, global_expanded], dim=-1)
        return self.fusion_mlp(fused)


# ============================================================
# 实现3: GNNEncoder — 图神经网络编码器
# ============================================================
class GNNEncoder(GeometricEncoder):
    """
    图神经网络编码器 —— 利用网格拓扑进行消息传递。

    通过 PyTorch Geometric 的图卷积层在 FEM 网格上传递信息，
    使每个节点的表示融合其邻居的几何和物理特征。
    这对振动分析尤其重要，因为应力/应变波沿网格单元传播。

    支持的卷积类型:
        - 'gcn':  GCNConv — 图卷积网络 (Kipf & Welling)
        - 'sage': SAGEConv — GraphSAGE (Hamilton et al.)
        - 'gat':  GATConv — 图注意力网络 (Velickovic et al.)
        - 'gin':  GINConv — 图同构网络 (Xu et al.)

    无拓扑时的回退:
        若 geometry_data.edge_index 为 None，自动回退为 SimplePointEncoder (纯MLP)。

    参考:
        - Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks"
        - Hamilton et al., "Inductive Representation Learning on Large Graphs"
        - Velickovic et al., "Graph Attention Networks"
    """

    def __init__(self, in_channels: int = 0, hidden_dim: int = 256, out_dim: int = 256,
                 coord_dim: int = 3, n_layers: int = 3,
                 conv_type: str = 'sage',
                 # GAT 特定参数
                 gat_heads: int = 4,
                 gat_dropout: float = 0.0,
                 # 全局特征
                 use_global_pool: bool = True,
                 global_pool: str = 'mean'):
        """
        参数:
            in_channels:     每点附加特征维度 (如材料参数)
            hidden_dim:      隐藏层宽度
            out_dim:         输出逐点特征维度
            coord_dim:       坐标维度
            n_layers:        图卷积层数
            conv_type:       卷积类型: 'gcn' | 'sage' | 'gat' | 'gin'
            gat_heads:       GAT 注意力头数 (仅 conv_type='gat')
            gat_dropout:     GAT 注意力 dropout
            use_global_pool: 是否将全局池化特征拼接到每个节点
            global_pool:     全局池化方式: 'mean' | 'max' | 'meanmax'
        """
        super().__init__(in_channels, hidden_dim, out_dim)
        self.coord_dim = coord_dim
        self.n_layers = n_layers
        self.conv_type = conv_type
        self.use_global_pool = use_global_pool
        self.global_pool_type = global_pool
        self.gat_heads = gat_heads

        input_dim = coord_dim + in_channels

        # 输入投影: 原始特征 → 隐空间
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
        )

        # 构建图卷积层 (直接创建 PyG 层)
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(
                self._create_conv(hidden_dim, hidden_dim, conv_type, gat_heads, gat_dropout)
            )

        # 全局池化后的融合 MLP (可选)
        if use_global_pool:
            global_in = hidden_dim * 2 if global_pool == 'meanmax' else hidden_dim
            self.global_fusion = nn.Sequential(
                nn.Linear(hidden_dim + global_in, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.output_proj = nn.Linear(hidden_dim, out_dim)

        # 无拓扑时的回退 MLP
        self.fallback_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    @staticmethod
    def _create_conv(in_dim, out_dim, conv_type, gat_heads, gat_dropout):
        """创建单个图卷积层"""
        from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv

        if conv_type == 'gcn':
            return GCNConv(in_dim, out_dim)
        elif conv_type == 'sage':
            return SAGEConv(in_dim, out_dim)
        elif conv_type == 'gat':
            return GATConv(in_dim, out_dim // gat_heads,
                          heads=gat_heads, dropout=gat_dropout)
        elif conv_type == 'gin':
            mlp = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LeakyReLU(0.1),
                nn.Linear(out_dim, out_dim),
            )
            return GINConv(mlp)
        else:
            raise ValueError(f"未知卷积类型: {conv_type}，可选 'gcn'|'sage'|'gat'|'gin'")

    def _global_pool_fn(self, x, batch):
        """全局池化: (total_N, D) → 扩展到每个节点 (total_N, D_pool)"""
        if batch is None:
            # 无 batch 索引时，对整个张量做池化
            pooled = x.mean(dim=0, keepdim=True)  # (1, D)
            return pooled.expand(x.shape[0], -1)   # (total_N, D)

        from torch_geometric.nn import global_mean_pool, global_max_pool

        if self.global_pool_type == 'mean':
            pooled = global_mean_pool(x, batch)
        elif self.global_pool_type == 'max':
            pooled = global_max_pool(x, batch)
        elif self.global_pool_type == 'meanmax':
            pooled = torch.cat([
                global_mean_pool(x, batch),
                global_max_pool(x, batch),
            ], dim=-1)
        else:
            raise ValueError(f"未知池化方式: {self.global_pool_type}")

        return pooled[batch]  # (total_N, D_pool)

    def forward(self, geometry_data: GeometryData) -> torch.Tensor:
        """
        GNN 编码流程:
            1. 坐标 + 特征拼接 → 输入投影 → (total_N, hidden_dim)
            2. 多层图卷积消息传递 (如果有 edge_index)
            3. (可选) 全局池化特征拼接到每个节点
            4. 输出投影 → 恢复 (B, N, out_dim) 形状

        无拓扑时回退为 MLP。
        """
        points = geometry_data.points
        point_feat = geometry_data.point_features
        edge_index = geometry_data.edge_index
        batch_idx = geometry_data.batch

        # 处理 edge_index: 单样本时可能是 (1, 2, E) 需要 squeeze 为 (2, E)
        if edge_index is not None and edge_index.ndim == 3:
            edge_index = edge_index.squeeze(0)  # (1, 2, E) → (2, E)

        # 确定形状
        if points.ndim == 3:
            B, N, _ = points.shape
            points_flat = points.reshape(B * N, -1)
            if point_feat is not None:
                point_feat_flat = point_feat.reshape(B * N, -1)
            else:
                point_feat_flat = None
        else:
            B = None
            points_flat = points
            point_feat_flat = point_feat

        # 拼接特征
        if point_feat_flat is not None:
            x = torch.cat([points_flat, point_feat_flat], dim=-1)
        else:
            x = points_flat

        # —— 无拓扑 → 回退 MLP ——
        if edge_index is None:
            out_flat = self.fallback_mlp(x)
            if B is not None:
                return out_flat.view(B, N, -1)
            return out_flat

        # —— 有拓扑 → GNN 消息传递 ——
        x = self.input_proj(x)  # (total_N, hidden_dim)

        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.nn.functional.leaky_relu(x, 0.1)  # LeakyReLU: 保留负值信号, 适配振动模态的正负变化

        # 可选: 拼接全局特征
        if self.use_global_pool:
            global_feat = self._global_pool_fn(x, batch_idx)
            x = torch.cat([x, global_feat], dim=-1)
            x = self.global_fusion(x)
        else:
            x = self.output_proj(x)

        # 恢复形状
        if B is not None:
            x = x.view(B, N, -1)

        return x  # (B, N, out_dim) 或 (total_N, out_dim)


# ============================================================
# 实现4: DeepONetEncoder — Branch/Trunk 解耦架构
# ============================================================
class DeepONetEncoder(GeometricEncoder):
    """
    DeepONet 风格几何编码器 —— 解耦宏观特征与空间坐标。

    受 Geom-DeepONet 启发，将编码过程分为两个独立网络:

    Branch (宏观编码):
        处理非空间的宏观点特征 (材料参数、几何尺寸等),
        通过 MLP + 全局池化 → 宏观隐变量 V_macro (B, D)

    Trunk (空间编码):
        纯粹编码空间坐标 (x, y, z)，使用 SIREN 激活函数,
        精确捕捉高频空间梯度 → 逐点空间编码 (B, N, D)

    融合:
        Option "modulate": spatial * (1 + macro_expanded)   [FiLM风格]
        Option "dot":      sum(spatial * macro_expanded, dim=-1) → MLP → out_dim

    参考:
        - Lu et al., "DeepONet: Learning nonlinear operators"
        - Li et al., "Geom-DeepONet: Learning operators on geometric domains"
        - Sitzmann et al., "SIREN: Implicit Neural Representations with Periodic Activations"
    """

    def __init__(self, in_channels: int = 0, hidden_dim: int = 256, out_dim: int = 256,
                 coord_dim: int = 3,
                 # Branch 参数
                 branch_layers: int = 3,
                 branch_pool: str = "mean",
                 # Trunk (SIREN) 参数
                 trunk_layers: int = 4,
                 siren_w0: float = 30.0,
                 # 融合参数
                 fusion: str = "modulate",
                 # 无 point_features 时的 fallback
                 use_coord_branch: bool = True):
        """
        参数:
            in_channels:      每点的附加宏观特征维度 (如材料参数)
            hidden_dim:       隐藏层宽度
            out_dim:          输出逐点特征维度
            coord_dim:        坐标维度 (2 或 3)

            branch_layers:    Branch MLP层数 (处理宏观特征)
            branch_pool:      全局池化方式: "mean" | "max" | "meanmax"
            trunk_layers:     Trunk SIREN 层数 (处理空间坐标)
            siren_w0:         SIREN 频率因子 (越大→捕捉越高频的空间变化)
            fusion:           融合方式: "modulate" | "dot"
            use_coord_branch: 当无 point_features 时，是否用坐标的全局池化作为 branch 输入
        """
        super().__init__(in_channels, hidden_dim, out_dim)
        self.fusion = fusion
        self.branch_pool = branch_pool
        self.use_coord_branch = use_coord_branch

        # ========================
        # Branch: 宏观特征编码
        # ========================
        # 输入: point_features (B, N, in_channels) 或坐标投影
        # 输出: macro_latent (B, out_dim)

        # 确定 branch_net 的实际输入维度
        if in_channels > 0:
            # 有显式宏观特征 → 直接使用
            branch_in = in_channels
            self.coord_proj = None
        elif use_coord_branch:
            # 无宏观特征 → 对坐标做投影后作为 branch 输入
            branch_in = hidden_dim  # coord_proj 输出的维度
            self.coord_proj = nn.Sequential(
                nn.Linear(coord_dim, hidden_dim),
                nn.ReLU(),
            )
        else:
            # 完全无特征 → 使用可学习全局 token
            branch_in = hidden_dim
            self.coord_proj = None

        branch_layers_list = []
        current_dim = branch_in
        for i in range(branch_layers):
            if i == branch_layers - 1:
                branch_layers_list.append(nn.Linear(current_dim, out_dim))
            else:
                branch_layers_list.append(nn.Linear(current_dim, hidden_dim))
                branch_layers_list.append(nn.ReLU())
                current_dim = hidden_dim
        self.branch_net = nn.Sequential(*branch_layers_list)

        # ========================
        # Trunk: 空间坐标编码 (SIREN)
        # ========================
        # 输入: points (B, N, coord_dim)
        # 输出: spatial_encoding (B, N, out_dim)
        self.trunk_net = SirenMLP(
            in_dim=coord_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            n_layers=trunk_layers,
            w0=siren_w0,
        )

        # ========================
        # Fusion
        # ========================
        if fusion == "dot":
            # dot product 后接一个小 MLP
            self.fusion_mlp = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
        elif fusion == "modulate":
            # FiLM 风格调制: spatial * (1 + alpha * macro)
            self.macro_scale = nn.Parameter(torch.ones(1) * 0.1)
        else:
            raise ValueError(f"未知融合方式: {fusion}，可选 'modulate' | 'dot'")

    def _get_branch_input(self, geometry_data: GeometryData) -> torch.Tensor:
        """获取 Branch 网络的输入特征 (B, N, branch_in)"""
        if geometry_data.point_features is not None:
            return geometry_data.point_features  # (B, N, in_channels)
        elif self.coord_proj is not None:
            # 无宏观特征时，对坐标编码作为 fallback
            return self.coord_proj(geometry_data.points)  # (B, N, hidden_dim)
        else:
            # 完全无特征 → 使用常数输入 + 全局池化会退化为可学习偏置
            B, N, _ = geometry_data.points.shape
            return torch.ones(B, N, self.in_channels or self.hidden_dim,
                            device=geometry_data.points.device)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """全局池化: (B, N, D) → (B, D)"""
        if self.branch_pool == "mean":
            return x.mean(dim=1)
        elif self.branch_pool == "max":
            return x.max(dim=1)[0]
        elif self.branch_pool == "meanmax":
            return torch.cat([x.mean(dim=1), x.max(dim=1)[0]], dim=-1)
        else:
            raise ValueError(f"未知池化方式: {self.branch_pool}")

    def forward(self, geometry_data: GeometryData) -> torch.Tensor:
        """
        DeepONet 编码流程:
            1. Branch: 宏观特征 → pool → macro_latent (B, out_dim)
            2. Trunk: 空间坐标 → SIREN → spatial_enc (B, N, out_dim)
            3. Fusion: modulate 或 dot product

        返回:
            per_point_features: (B, N, out_dim)
        """
        points = geometry_data.points  # (B, N, 3)

        # —— Branch: 宏观特征 → 全局隐变量 ——
        branch_input = self._get_branch_input(geometry_data)  # (B, N, branch_in)
        branch_feat = self.branch_net(branch_input)            # (B, N, out_dim)
        macro = self._pool(branch_feat)                       # (B, out_dim)

        # —— Trunk: 空间坐标 → SIREN 逐点空间编码 ——
        spatial = self.trunk_net(points)  # (B, N, out_dim)

        # —— Fusion ——
        if self.fusion == "modulate":
            # 空间特征被宏观特征调制: spatial *= (1 + scale * macro)
            # macro_expanded: (B, 1, out_dim)
            modulation = 1.0 + self.macro_scale * macro.unsqueeze(1)
            out = spatial * modulation

        elif self.fusion == "dot":
            # DeepONet 经典点积: sum(spatial * macro, dim=-1) → MLP
            dot = (spatial * macro.unsqueeze(1)).sum(dim=-1, keepdim=True)  # (B, N, 1)
            out = self.fusion_mlp(dot)  # (B, N, out_dim)

        return out
