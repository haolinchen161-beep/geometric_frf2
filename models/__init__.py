"""
models/ — 几何→FRF 模型模块

导出:
    - Film:                   FiLM 条件调制层
    - GeometryData:           统一几何数据容器
    - Sine / SirenLayer / SirenMLP: SIREN 正弦激活网络
    - GeometricEncoder:       编码器抽象基类
    - SimplePointEncoder:     最简逐点MLP编码器
    - PointNetEncoder:        PointNet风格编码器
    - GNNEncoder:             GNN编码器 (预留)
    - DeepONetEncoder:        Branch/Trunk解耦架构 (SIREN空间编码)
    - PerPointFRFDecoder:     逐点FRF解码器 (分块, 支持复数)
    - PerPointFRFDecoderSimple: 逐点FRF解码器 (无分块)
    - GeometricFRFModel:      完整几何→FRF模型
    - build_geometric_model:  便捷构建函数
"""

from .film import Film
from .geometry_data import GeometryData
from .siren import Sine, SirenLayer, SirenMLP
from .geometric_encoder import (
    GeometricEncoder,
    SimplePointEncoder,
    PointNetEncoder,
    GNNEncoder,
    DeepONetEncoder,
)
from .perpoint_decoder import (
    PerPointFRFDecoder,
    PerPointFRFDecoderSimple,
)
from .frf_model import (
    GeometricFRFModel,
    build_geometric_model,
)
