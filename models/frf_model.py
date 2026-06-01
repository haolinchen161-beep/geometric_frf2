"""
frf_model.py — 模型构建入口。
"""
from .modal_model import ModalFRFModel


def build_geometric_model(encoder_type='modal',
                          encoder_kwargs=None, decoder_kwargs=None):
    """构建模型。encoder_type='modal' → ModalFRFModel。"""
    if encoder_type == 'modal':
        return ModalFRFModel(**(encoder_kwargs or {}), **(decoder_kwargs or {}))
    raise ValueError(f"未知编码器类型: {encoder_type}")
