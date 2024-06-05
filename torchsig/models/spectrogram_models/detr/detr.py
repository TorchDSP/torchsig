import os.path
from typing import Dict

import gdown
import numpy as np
import timm
import torch
from torch import nn

from .modules import *
from .utils import *

__all__ = [
    "detr_b0_nano",
    "detr_b2_nano",
    "detr_b4_nano",
    "detr_b0_nano_mod_family",
    "detr_b2_nano_mod_family",
    "detr_b4_nano_mod_family",
]

model_urls: Dict[str, str] = {
    "detr_b0_nano": "1t6V3M5hJC8C-RSwPtgKGG89u5doibs46",
    "detr_b2_nano": "1voDx7e0pBe_lGa_1sUYG8gyzOqz8nxmw",
    "detr_b4_nano": "1RA7yGvpKiIXHXl_o89Zn6R2dVVTgKsWO",
    "detr_b0_nano_mod_family": "1w42OxyAFf7CTJ5Yw8OU-kAZQZCpkNyaz",
    "detr_b2_nano_mod_family": "1Wd8QD5Eq2mbEz3hkMlAQFxWZcxZChLma",
    "detr_b4_nano_mod_family": "1ykrztgBc6c9knk1F2OirSUE_W3YbsTdB",
}


def detr_b0_nano(
    pretrained: bool = False,
    path: str = "detr_b0_nano.pt",
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
        pretrained (bool): If True, returns a model pre-trained on WBSig53
        path (str): Path to existing model or where to download checkpoint to
        num_classes (int): Number of output classes; if loading checkpoint and number does not equal 1, final layer will not be loaded from checkpoint
        drop_path_rate_backbone (float): Backbone drop path rate for training
        drop_rate_backbone (float): Backbone dropout rate for training
        drop_path_rate_transformer (float): Transformer drop path rate for training

    """
    # Create DETR-B0-Nano
    mdl = create_detr(
        backbone="efficientnet_b0",
        transformer="xcit-nano",
        num_classes=1,
        num_objects=50,
        hidden_dim=256,
        drop_rate_backbone=drop_rate_backbone,
        drop_path_rate_backbone=drop_path_rate_backbone,
        drop_path_rate_transformer=drop_path_rate_transformer,
        ds_rate_transformer=2,
        ds_method_transformer="chunker",
    )
    if pretrained:
        model_exists = os.path.exists(path)
        if not model_exists:
            file_id = model_urls["detr_b0_nano"]
            dl = gdown.download(id=file_id, output=path)
        mdl.load_state_dict(torch.load(path), strict=False)
    if num_classes != 1:
        mdl.linear_class = nn.Linear(
            mdl.linear_class.in_features,  # type: ignore
            num_classes,
        )
    return mdl


def detr_b2_nano(
    pretrained: bool = False,
    path: str = "detr_b2_nano.pt",
    num_classes: int = 1,
    drop_rate_backbone: float = 0.3,
    drop_path_rate_backbone: float = 0.2,
    drop_path_rate_transformer: float = 0.1,
):
    """Constructs a DETR architecture with an EfficientNet-B2 backbone and an XCiT-Nano transformer.
    DETR from `"End-to-End Object Detection with Transformers" <https://arxiv.org/pdf/2005.12872.pdf>`_.
    EfficientNet from `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/pdf/1905.11946.pdf>`_.
    XCiT from `"XCiT: Cross-Covariance Image Transformers" <https://arxiv.org/pdf/2106.09681.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on WBSig53
        path (str): Path to existing model or where to download checkpoint to
        num_classes (int): Number of output classes; if loading checkpoint and number does not equal 1, final layer will not be loaded from checkpoint
        drop_path_rate_backbone (float): Backbone drop path rate for training
        drop_rate_backbone (float): Backbone dropout rate for training
        drop_path_rate_transformer (float): Transformer drop path rate for training

    """
    # Create DETR-B2-Nano
    mdl = create_detr(
        backbone="efficientnet_b2",
        transformer="xcit-nano",
        num_classes=1,
        num_objects=50,
        hidden_dim=256,
        drop_rate_backbone=drop_rate_backbone,
        drop_path_rate_backbone=drop_path_rate_backbone,
        drop_path_rate_transformer=drop_path_rate_transformer,
        ds_rate_transformer=2,
        ds_method_transformer="chunker",
    )
    if pretrained:
        model_exists = os.path.exists(path)
        if not model_exists:
            file_id = model_urls["detr_b2_nano"]
            dl = gdown.download(id=file_id, output=path)
        mdl.load_state_dict(torch.load(path), strict=False)
    if num_classes != 1:
        mdl.linear_class = nn.Linear(
            mdl.linear_class.in_features,  # type: ignore
            num_classes,
        )
    return mdl


def detr_b4_nano(
    pretrained: bool = False,
    path: str = "detr_b4_nano.pt",
    num_classes: int = 1,
    drop_rate_backbone: float = 0.4,
    drop_path_rate_backbone: float = 0.2,
    drop_path_rate_transformer: float = 0.1,
):
    """Constructs a DETR architecture with an EfficientNet-B4 backbone and an XCiT-Nano transformer.
    DETR from `"End-to-End Object Detection with Transformers" <https://arxiv.org/pdf/2005.12872.pdf>`_.
    EfficientNet from `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/pdf/1905.11946.pdf>`_.
    XCiT from `"XCiT: Cross-Covariance Image Transformers" <https://arxiv.org/pdf/2106.09681.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on WBSig53
        path (str): Path to existing model or where to download checkpoint to
        num_classes (int): Number of output classes; if loading checkpoint and number does not equal 1, final layer will not be loaded from checkpoint
        drop_path_rate_backbone (float): Backbone drop path rate for training
        drop_rate_backbone (float): Backbone dropout rate for training
        drop_path_rate_transformer (float): Transformer drop path rate for training

    """
    # Create DETR-B4-Nano
    mdl = create_detr(
        backbone="efficientnet_b4",
        transformer="xcit-nano",
        num_classes=1,
        num_objects=50,
        hidden_dim=256,
        drop_rate_backbone=drop_rate_backbone,
        drop_path_rate_backbone=drop_path_rate_backbone,
        drop_path_rate_transformer=drop_path_rate_transformer,
        ds_rate_transformer=2,
        ds_method_transformer="chunker",
    )
    if pretrained:
        model_exists = os.path.exists(path)
        if not model_exists:
            file_id = model_urls["detr_b4_nano"]
            dl = gdown.download(id=file_id, output=path)
        mdl.load_state_dict(torch.load(path), strict=False)
    if num_classes != 1:
        mdl.linear_class = nn.Linear(
            mdl.linear_class.in_features,  # type: ignore
            num_classes,
        )
    return mdl


def detr_b0_nano_mod_family(
    pretrained: bool = False,
    path: str = "detr_b0_nano_mod_family.pt",
    num_classes: int = 6,
    drop_rate_backbone: float = 0.2,
    drop_path_rate_backbone: float = 0.2,
    drop_path_rate_transformer: float = 0.1,
):
    """Constructs a DETR architecture with an EfficientNet-B0 backbone and an XCiT-Nano transformer.
    DETR from `"End-to-End Object Detection with Transformers" <https://arxiv.org/pdf/2005.12872.pdf>`_.
    EfficientNet from `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/pdf/1905.11946.pdf>`_.
    XCiT from `"XCiT: Cross-Covariance Image Transformers" <https://arxiv.org/pdf/2106.09681.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on WBSig53
        path (str): Path to existing model or where to download checkpoint to
        num_classes (int): Number of output classes; if loading checkpoint and number does not equal 6, final layer will not be loaded from checkpoint
        drop_path_rate_backbone (float): Backbone drop path rate for training
        drop_rate_backbone (float): Backbone dropout rate for training
        drop_path_rate_transformer (float): Transformer drop path rate for training

    """
    # Create DETR-B0-Nano
    mdl = create_detr(
        backbone="efficientnet_b0",
        transformer="xcit-nano",
        num_classes=6,
        num_objects=50,
        hidden_dim=256,
        drop_rate_backbone=drop_rate_backbone,
        drop_path_rate_backbone=drop_path_rate_backbone,
        drop_path_rate_transformer=drop_path_rate_transformer,
        ds_rate_transformer=2,
        ds_method_transformer="chunker",
    )
    if pretrained:
        model_exists = os.path.exists(path)
        if not model_exists:
            file_id = model_urls["detr_b0_nano_mod_family"]
            dl = gdown.download(id=file_id, output=path)
        mdl.load_state_dict(torch.load(path), strict=False)
    if num_classes != 6:
        mdl.linear_class = nn.Linear(
            mdl.linear_class.in_features,  # type: ignore
            num_classes,
        )
    return mdl


def detr_b2_nano_mod_family(
    pretrained: bool = False,
    path: str = "detr_b2_nano_mod_family.pt",
    num_classes: int = 1,
    drop_rate_backbone: float = 0.3,
    drop_path_rate_backbone: float = 0.2,
    drop_path_rate_transformer: float = 0.1,
):
    """Constructs a DETR architecture with an EfficientNet-B2 backbone and an XCiT-Nano transformer.
    DETR from `"End-to-End Object Detection with Transformers" <https://arxiv.org/pdf/2005.12872.pdf>`_.
    EfficientNet from `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/pdf/1905.11946.pdf>`_.
    XCiT from `"XCiT: Cross-Covariance Image Transformers" <https://arxiv.org/pdf/2106.09681.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on WBSig53
        path (str): Path to existing model or where to download checkpoint to
        num_classes (int): Number of output classes; if loading checkpoint and number does not equal 6, final layer will not be loaded from checkpoint
        drop_path_rate_backbone (float): Backbone drop path rate for training
        drop_rate_backbone (float): Backbone dropout rate for training
        drop_path_rate_transformer (float): Transformer drop path rate for training

    """
    # Create DETR-B2-Nano
    mdl = create_detr(
        backbone="efficientnet_b2",
        transformer="xcit-nano",
        num_classes=6,
        num_objects=50,
        hidden_dim=256,
        drop_rate_backbone=drop_rate_backbone,
        drop_path_rate_backbone=drop_path_rate_backbone,
        drop_path_rate_transformer=drop_path_rate_transformer,
        ds_rate_transformer=2,
        ds_method_transformer="chunker",
    )
    if pretrained:
        model_exists = os.path.exists(path)
        if not model_exists:
            file_id = model_urls["detr_b2_nano_mod_family"]
            dl = gdown.download(id=file_id, output=path)
        mdl.load_state_dict(torch.load(path), strict=False)
    if num_classes != 6:
        mdl.linear_class = nn.Linear(
            mdl.linear_class.in_features,  # type: ignore
            num_classes,
        )
    return mdl


def detr_b4_nano_mod_family(
    pretrained: bool = False,
    path: str = "detr_b4_nano_mod_family.pt",
    num_classes: int = 6,
    drop_rate_backbone: float = 0.4,
    drop_path_rate_backbone: float = 0.2,
    drop_path_rate_transformer: float = 0.1,
):
    """Constructs a DETR architecture with an EfficientNet-B4 backbone and an XCiT-Nano transformer.
    DETR from `"End-to-End Object Detection with Transformers" <https://arxiv.org/pdf/2005.12872.pdf>`_.
    EfficientNet from `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/pdf/1905.11946.pdf>`_.
    XCiT from `"XCiT: Cross-Covariance Image Transformers" <https://arxiv.org/pdf/2106.09681.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on WBSig53
        path (str): Path to existing model or where to download checkpoint to
        num_classes (int): Number of output classes; if loading checkpoint and number does not equal 6, final layer will not be loaded from checkpoint
        drop_path_rate_backbone (float): Backbone drop path rate for training
        drop_rate_backbone (float): Backbone dropout rate for training
        drop_path_rate_transformer (float): Transformer drop path rate for training

    """
    # Create DETR-B4-Nano
    mdl = create_detr(
        backbone="efficientnet_b4",
        transformer="xcit-nano",
        num_classes=6,
        num_objects=50,
        hidden_dim=256,
        drop_rate_backbone=drop_rate_backbone,
        drop_path_rate_backbone=drop_path_rate_backbone,
        drop_path_rate_transformer=drop_path_rate_transformer,
        ds_rate_transformer=2,
        ds_method_transformer="chunker",
    )
    if pretrained:
        model_exists = os.path.exists(path)
        if not model_exists:
            file_id = model_urls["detr_b0_nano_mod_family"]
            dl = gdown.download(id=file_id, output=path)
        mdl.load_state_dict(torch.load(path), strict=False)
    if num_classes != 6:
        mdl.linear_class = nn.Linear(
            mdl.linear_class.in_features,  # type: ignore
            num_classes,
        )
    return mdl
