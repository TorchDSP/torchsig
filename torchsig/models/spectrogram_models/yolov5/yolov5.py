import timm
import gdown
import torch
import os.path
import numpy as np
from torch import nn

from .modules import *
from .utils import *
from .mean_ap import *


__all__ = [
    "yolov5p", "yolov5n", "yolov5s",
    "yolov5p_mod_family", "yolov5n_mod_family", "yolov5s_mod_family",
]

model_urls = {
    "yolov5p": "1d1ihKbtGQciRwqmBDrHiESZ22zx9W01S",
    "yolov5n": "184h1f8-DV3FDYd01X7TdKmxWZ2s73FiH",
    "yolov5s": "1t7hHB4uXJ0BaSEmq_li2oj1tEDXgZh0z",
    "yolov5p_mod_family": "1z8VLEpVqQEFPW3u4T3Yd6c5J0e__UDqf",
    "yolov5n_mod_family": "1B2ke51DGbpZXOMhuWTQLXZaDM59VC5Mm",
    "yolov5s_mod_family": "1HzcKfM4URtAqhCIQr_obXWWbYIFsEE4s",
}


def yolov5p(
    pretrained: bool = False, 
    path: str = "yolov5p.pt",
    num_classes: int = 1,
):
    """Constructs a YOLOv5 architecture with Pico scaling.
    YOLOv5 from `"YOLOv5 GitHub" <https://github.com/ultralytics/yolov5>`_.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on WBSig53
        path (str): Path to existing model or where to download checkpoint to
        num_classes (int): Number of output classes; if loading checkpoint and number does not equal 1, final layer will not be loaded from checkpoint
        
    """
    # Create YOLOv5p
    mdl = create_yolov5(
        network='yolov5p',
        num_classes=1,
    )
    if pretrained:
        model_exists = os.path.exists(path)
        if not model_exists:
            file_id = model_urls['yolov5p']
            dl = gdown.download(id=file_id, output=path)
        mdl.load_state_dict(torch.load(path), strict=False)
    if num_classes != 1:
        mdl.model[-1].no = int(mdl.model[-1].no / (1 + 5) * (num_classes + 5))
        for det_conv_idx in range(len(mdl.model[-1].m)):
            mdl.model[-1].m[det_conv_idx] = torch.nn.Conv2d(
                in_channels=mdl.model[-1].m[det_conv_idx].in_channels,
                out_channels=int(mdl.model[-1].m[det_conv_idx].out_channels / (1+5) * (num_classes + 5)),
                kernel_size=mdl.model[-1].m[det_conv_idx].kernel_size,
                stride=mdl.model[-1].m[det_conv_idx].stride,
            )
    return mdl
    
    
def yolov5n(
    pretrained: bool = False, 
    path: str = "yolov5n.pt",
    num_classes: int = 1,
):
    """Constructs a YOLOv5 architecture with Nano scaling.
    YOLOv5 from `"YOLOv5 GitHub" <https://github.com/ultralytics/yolov5>`_.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on WBSig53
        path (str): Path to existing model or where to download checkpoint to
        num_classes (int): Number of output classes; if loading checkpoint and number does not equal 1, final layer will not be loaded from checkpoint
        
    """
    # Create YOLOv5p
    mdl = create_yolov5(
        network='yolov5n',
        num_classes=1,
    )
    if pretrained:
        model_exists = os.path.exists(path)
        if not model_exists:
            file_id = model_urls['yolov5n']
            dl = gdown.download(id=file_id, output=path)
        mdl.load_state_dict(torch.load(path), strict=False)
    if num_classes != 1:
        mdl.model[-1].no = int(mdl.model[-1].no / (1 + 5) * (num_classes + 5))
        for det_conv_idx in range(len(mdl.model[-1].m)):
            mdl.model[-1].m[det_conv_idx] = torch.nn.Conv2d(
                in_channels=mdl.model[-1].m[det_conv_idx].in_channels,
                out_channels=int(mdl.model[-1].m[det_conv_idx].out_channels / (1+5) * (num_classes + 5)),
                kernel_size=mdl.model[-1].m[det_conv_idx].kernel_size,
                stride=mdl.model[-1].m[det_conv_idx].stride,
            )    
    return mdl


def yolov5s(
    pretrained: bool = False, 
    path: str = "yolov5s.pt",
    num_classes: int = 1,
):
    """Constructs a YOLOv5 architecture with Small scaling.
    YOLOv5 from `"YOLOv5 GitHub" <https://github.com/ultralytics/yolov5>`_.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on WBSig53
        path (str): Path to existing model or where to download checkpoint to
        num_classes (int): Number of output classes; if loading checkpoint and number does not equal 1, final layer will not be loaded from checkpoint
        
    """
    # Create YOLOv5p
    mdl = create_yolov5(
        network='yolov5s',
        num_classes=1,
    )
    if pretrained:
        model_exists = os.path.exists(path)
        if not model_exists:
            file_id = model_urls['yolov5s']
            dl = gdown.download(id=file_id, output=path)
        mdl.load_state_dict(torch.load(path), strict=False)
    if num_classes != 1:
        mdl.model[-1].no = int(mdl.model[-1].no / (1 + 5) * (num_classes + 5))
        for det_conv_idx in range(len(mdl.model[-1].m)):
            mdl.model[-1].m[det_conv_idx] = torch.nn.Conv2d(
                in_channels=mdl.model[-1].m[det_conv_idx].in_channels,
                out_channels=int(mdl.model[-1].m[det_conv_idx].out_channels / (1+5) * (num_classes + 5)),
                kernel_size=mdl.model[-1].m[det_conv_idx].kernel_size,
                stride=mdl.model[-1].m[det_conv_idx].stride,
            )
    return mdl


def yolov5p_mod_family(
    pretrained: bool = False, 
    path: str = "yolov5p.pt",
    num_classes: int = 6,
):
    """Constructs a YOLOv5 architecture with Pico scaling.
    YOLOv5 from `"YOLOv5 GitHub" <https://github.com/ultralytics/yolov5>`_.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on WBSig53
        path (str): Path to existing model or where to download checkpoint to
        num_classes (int): Number of output classes; if loading checkpoint and number does not equal 1, final layer will not be loaded from checkpoint
        
    """
    # Create YOLOv5p
    mdl = create_yolov5(
        network='yolov5p',
        num_classes=6,
    )
    if pretrained:
        model_exists = os.path.exists(path)
        if not model_exists:
            file_id = model_urls['yolov5p_mod_family']
            dl = gdown.download(id=file_id, output=path)
        mdl.load_state_dict(torch.load(path), strict=False)
    if num_classes != 6:
        mdl.model[-1].no = int(mdl.model[-1].no / (6 + 5) * (num_classes + 5))
        for det_conv_idx in range(len(mdl.model[-1].m)):
            mdl.model[-1].m[det_conv_idx] = torch.nn.Conv2d(
                in_channels=mdl.model[-1].m[det_conv_idx].in_channels,
                out_channels=int(mdl.model[-1].m[det_conv_idx].out_channels / (6+5) * (num_classes + 5)),
                kernel_size=mdl.model[-1].m[det_conv_idx].kernel_size,
                stride=mdl.model[-1].m[det_conv_idx].stride,
            )
    return mdl
    
    
def yolov5n_mod_family(
    pretrained: bool = False, 
    path: str = "yolov5n.pt",
    num_classes: int = 6,
):
    """Constructs a YOLOv5 architecture with Nano scaling.
    YOLOv5 from `"YOLOv5 GitHub" <https://github.com/ultralytics/yolov5>`_.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on WBSig53
        path (str): Path to existing model or where to download checkpoint to
        num_classes (int): Number of output classes; if loading checkpoint and number does not equal 1, final layer will not be loaded from checkpoint
        
    """
    # Create YOLOv5p
    mdl = create_yolov5(
        network='yolov5n',
        num_classes=6,
    )
    if pretrained:
        model_exists = os.path.exists(path)
        if not model_exists:
            file_id = model_urls['yolov5n_mod_family']
            dl = gdown.download(id=file_id, output=path)
        mdl.load_state_dict(torch.load(path), strict=False)
    if num_classes != 6:
        mdl.model[-1].no = int(mdl.model[-1].no / (6 + 5) * (num_classes + 5))
        for det_conv_idx in range(len(mdl.model[-1].m)):
            mdl.model[-1].m[det_conv_idx] = torch.nn.Conv2d(
                in_channels=mdl.model[-1].m[det_conv_idx].in_channels,
                out_channels=int(mdl.model[-1].m[det_conv_idx].out_channels / (6+5) * (num_classes + 5)),
                kernel_size=mdl.model[-1].m[det_conv_idx].kernel_size,
                stride=mdl.model[-1].m[det_conv_idx].stride,
            )
    return mdl


def yolov5s_mod_family(
    pretrained: bool = False, 
    path: str = "yolov5s.pt",
    num_classes: int = 6,
):
    """Constructs a YOLOv5 architecture with Small scaling.
    YOLOv5 from `"YOLOv5 GitHub" <https://github.com/ultralytics/yolov5>`_.
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on WBSig53
        path (str): Path to existing model or where to download checkpoint to
        num_classes (int): Number of output classes; if loading checkpoint and number does not equal 1, final layer will not be loaded from checkpoint
        
    """
    # Create YOLOv5p
    mdl = create_yolov5(
        network='yolov5s',
        num_classes=6,
    )
    if pretrained:
        model_exists = os.path.exists(path)
        if not model_exists:
            file_id = model_urls['yolov5s_mod_family']
            dl = gdown.download(id=file_id, output=path)
        mdl.load_state_dict(torch.load(path), strict=False)
    if num_classes != 6:
        mdl.model[-1].no = int(mdl.model[-1].no / (6 + 5) * (num_classes + 5))
        for det_conv_idx in range(len(mdl.model[-1].m)):
            mdl.model[-1].m[det_conv_idx] = torch.nn.Conv2d(
                in_channels=mdl.model[-1].m[det_conv_idx].in_channels,
                out_channels=int(mdl.model[-1].m[det_conv_idx].out_channels / (6+5) * (num_classes + 5)),
                kernel_size=mdl.model[-1].m[det_conv_idx].kernel_size,
                stride=mdl.model[-1].m[det_conv_idx].stride,
            )
    return mdl
