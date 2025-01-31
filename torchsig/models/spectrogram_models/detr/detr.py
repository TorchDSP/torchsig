import os.path
from typing import Dict

import numpy as np
import timm
import torch
from torch import nn

from .modules import *
from .utils import *

__all__ = [
    "DETR"
]

supported_detr_models = [
    "detr_b0_nano",
    "detr_b2_nano",
    "detr_b4_nano",
]

def DETR(
    version: str,
    num_classes: int = 1,
    drop_rate_backbone: float = 0.2,
    drop_path_rate_backbone: float = 0.2,
    drop_path_rate_transformer: float = 0.1,
):
    """Constructs a DETR architecture with an EfficientNet-B0 backbone and an XCiT-Nano transformer.
    DETR from `"End-to-End Object Detection with Transformers" <https://arxiv.org/pdf/2005.12872.pdf>`_.
    EfficientNet from `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/pdf/1905.11946.pdf>`_.
    XCiT from `"XCiT: Cross-Covariance Image Transformers" <https://arxiv.org/pdf/2106.09681.pdf>`_.

    Args:
        version (str): Which DETR model to load.
        num_classes (int): Number of output classes; if loading checkpoint and number does not equal 1, final layer will not be loaded from checkpoint
        drop_path_rate_backbone (float): Backbone drop path rate for training
        drop_rate_backbone (float): Backbone dropout rate for training
        drop_path_rate_transformer (float): Transformer drop path rate for training

    """
    if not version in supported_detr_models:
        raise ValueError(f"Invalid DETR model: {version}")

    if "b0" in version:
        backbone = "efficientnet_b0"
    elif "b2" in version:
        backbone = "efficientnet_b2"
    elif "b4" in version:
        backbone = "efficientnet_b4"
    else:
        raise ValueError(f"Invalid DETR model: {version}")

    if "nano" in version:
        transformer = "xcit-nano"
    else:
        raise ValueError(f"Invalid DETR model: {version}")

    # Create DETR
    mdl = create_detr(
        backbone=backbone,
        transformer=transformer,
        num_classes=num_classes,
        num_objects=50,
        hidden_dim=256,
        drop_rate_backbone=drop_rate_backbone,
        drop_path_rate_backbone=drop_path_rate_backbone,
        drop_path_rate_transformer=drop_path_rate_transformer,
        ds_rate_transformer=2,
        ds_method_transformer="chunker",
    )

    return mdl