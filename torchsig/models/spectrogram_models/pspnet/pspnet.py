import timm
import gdown
import torch
import os.path
import numpy as np
from torch import nn

from .modules import *
from .utils import *

__all__ = [
    "pspnet_b0", "pspnet_b2", "pspnet_b4",
    "pspnet_b0_mod_family", "pspnet_b2_mod_family", "pspnet_b4_mod_family",
]

model_urls = {
    "pspnet_b0": "1dSxMHzfiiqH8uAbWLhOy4jOmIJCP2M35",
    "pspnet_b2": "1VnDPdByVMihn1LMVRsU9-_Ndbzvzybvz",
    "pspnet_b4": "13gLlx1sSi5t6njp6NnPsphDBN_yYvOu0",
    "pspnet_b0_mod_family": "1I1FF0lek3APmrTHakz7LhmTMNkKSPcxg",
    "pspnet_b2_mod_family": "1803E3cGMhi2QMmv-Yh27VgE438iheKyJ",
    "pspnet_b4_mod_family": "1T8xVV2AnZIeEWIjXe9MKGK7kxdDfBxKM",
}


def pspnet_b0(
    pretrained: bool = False, 
    path: str = "pspnet_b0.pt",
    num_classes: int = 1,
):
    """Constructs a PSPNet architecture with an EfficientNet-B0 backbone.
    PSPNet from `"Pyramid Scene Parsing Network" <https://arxiv.org/pdf/1612.01105.pdf>`_.
    EfficientNet from `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/pdf/1905.11946.pdf>`_.
    
    Args:
        pretrained (bool): 
            If True, returns a model pre-trained on WBSig53
        path (str): 
            Path to existing model or where to download checkpoint to
        num_classes (int): 
            Number of output classes; if loading checkpoint and 
            number does not equal 1, final layer will not be loaded from checkpoint
            NOTE: num_classes should equal the total number of classes **without**
            including the background class. That "class" is automatically included.
        
    """
    # Create PSPNet-B0
    mdl = create_pspnet(
        encoder='efficientnet-b0',
        num_classes=1+1,
    )
    if pretrained:
        model_exists = os.path.exists(path)
        if not model_exists:
            file_id = model_urls['pspnet_b0']
            dl = gdown.download(id=file_id, output=path)
        mdl.load_state_dict(torch.load(path), strict=False)
    if num_classes != 1:
        mdl.segmentation_head[0] = torch.nn.Conv2d(
            in_channels=mdl.segmentation_head[0].in_channels,
            out_channels=num_classes+1,
            kernel_size=mdl.segmentation_head[0].kernel_size,
            stride=mdl.segmentation_head[0].stride,
            padding=mdl.segmentation_head[0].padding,
        )
    return mdl


def pspnet_b2(
    pretrained: bool = False, 
    path: str = "pspnet_b2.pt",
    num_classes: int = 1,
):
    """Constructs a PSPNet architecture with an EfficientNet-B2 backbone.
    PSPNet from `"Pyramid Scene Parsing Network" <https://arxiv.org/pdf/1612.01105.pdf>`_.
    EfficientNet from `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/pdf/1905.11946.pdf>`_.
    
    Args:
        pretrained (bool): 
            If True, returns a model pre-trained on WBSig53
        path (str): 
            Path to existing model or where to download checkpoint to
        num_classes (int): 
            Number of output classes; if loading checkpoint and 
            number does not equal 1, final layer will not be loaded from checkpoint
            NOTE: num_classes should equal the total number of classes **without**
            including the background class. That "class" is automatically included.
        
    """
    # Create PSPNet-B2
    mdl = create_pspnet(
        encoder='efficientnet-b2',
        num_classes=1+1,
    )
    if pretrained:
        model_exists = os.path.exists(path)
        if not model_exists:
            file_id = model_urls['pspnet_b2']
            dl = gdown.download(id=file_id, output=path)
        mdl.load_state_dict(torch.load(path), strict=False)
    if num_classes != 1:
        mdl.segmentation_head[0] = torch.nn.Conv2d(
            in_channels=mdl.segmentation_head[0].in_channels,
            out_channels=num_classes+1,
            kernel_size=mdl.segmentation_head[0].kernel_size,
            stride=mdl.segmentation_head[0].stride,
            padding=mdl.segmentation_head[0].padding,
        )
    return mdl
    

def pspnet_b4(
    pretrained: bool = False, 
    path: str = "pspnet_b4.pt",
    num_classes: int = 1,
):
    """Constructs a PSPNet architecture with an EfficientNet-B4 backbone.
    PSPNet from `"Pyramid Scene Parsing Network" <https://arxiv.org/pdf/1612.01105.pdf>`_.
    EfficientNet from `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/pdf/1905.11946.pdf>`_.
    
    Args:
        pretrained (bool): 
            If True, returns a model pre-trained on WBSig53
        path (str): 
            Path to existing model or where to download checkpoint to
        num_classes (int): 
            Number of output classes; if loading checkpoint and 
            number does not equal 1, final layer will not be loaded from checkpoint
            NOTE: num_classes should equal the total number of classes **without**
            including the background class. That "class" is automatically included.
        
    """
    # Create PSPNet-B4
    mdl = create_pspnet(
        encoder='efficientnet-b4',
        num_classes=1+1,
    )
    if pretrained:
        model_exists = os.path.exists(path)
        if not model_exists:
            file_id = model_urls['pspnet_b4']
            dl = gdown.download(id=file_id, output=path)
        mdl.load_state_dict(torch.load(path), strict=False)
    if num_classes != 1:
        mdl.segmentation_head[0] = torch.nn.Conv2d(
            in_channels=mdl.segmentation_head[0].in_channels,
            out_channels=num_classes+1,
            kernel_size=mdl.segmentation_head[0].kernel_size,
            stride=mdl.segmentation_head[0].stride,
            padding=mdl.segmentation_head[0].padding,
        )
    return mdl


def pspnet_b0_mod_family(
    pretrained: bool = False, 
    path: str = "pspnet_b0_mod_family.pt",
    num_classes: int = 6,
):
    """Constructs a PSPNet architecture with an EfficientNet-B0 backbone.
    PSPNet from `"Pyramid Scene Parsing Network" <https://arxiv.org/pdf/1612.01105.pdf>`_.
    EfficientNet from `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/pdf/1905.11946.pdf>`_.
    
    Args:
        pretrained (bool): 
            If True, returns a model pre-trained on WBSig53
        path (str): 
            Path to existing model or where to download checkpoint to
        num_classes (int): 
            Number of output classes; if loading checkpoint and 
            number does not equal 1, final layer will not be loaded from checkpoint
            NOTE: num_classes should equal the total number of classes **without**
            including the background class. That "class" is automatically included.
        
    """
    # Create PSPNet-B0
    mdl = create_pspnet(
        encoder='efficientnet-b0',
        num_classes=6+1,
    )
    if pretrained:
        model_exists = os.path.exists(path)
        if not model_exists:
            file_id = model_urls['pspnet_b0_mod_family']
            dl = gdown.download(id=file_id, output=path)
        mdl.load_state_dict(torch.load(path), strict=False)
    if num_classes != 6:
        mdl.segmentation_head[0] = torch.nn.Conv2d(
            in_channels=mdl.segmentation_head[0].in_channels,
            out_channels=num_classes+1,
            kernel_size=mdl.segmentation_head[0].kernel_size,
            stride=mdl.segmentation_head[0].stride,
            padding=mdl.segmentation_head[0].padding,
        )
    return mdl


def pspnet_b2_mod_family(
    pretrained: bool = False, 
    path: str = "pspnet_b2_mod_family.pt",
    num_classes: int = 6,
):
    """Constructs a PSPNet architecture with an EfficientNet-B2 backbone.
    PSPNet from `"Pyramid Scene Parsing Network" <https://arxiv.org/pdf/1612.01105.pdf>`_.
    EfficientNet from `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/pdf/1905.11946.pdf>`_.
    
    Args:
        pretrained (bool): 
            If True, returns a model pre-trained on WBSig53
        path (str): 
            Path to existing model or where to download checkpoint to
        num_classes (int): 
            Number of output classes; if loading checkpoint and 
            number does not equal 1, final layer will not be loaded from checkpoint
            NOTE: num_classes should equal the total number of classes **without**
            including the background class. That "class" is automatically included.
        
    """
    # Create PSPNet-B2
    mdl = create_pspnet(
        encoder='efficientnet-b2',
        num_classes=6+1,
    )
    if pretrained:
        model_exists = os.path.exists(path)
        if not model_exists:
            file_id = model_urls['pspnet_b2_mod_family']
            dl = gdown.download(id=file_id, output=path)
        mdl.load_state_dict(torch.load(path), strict=False)
    if num_classes != 6:
        mdl.segmentation_head[0] = torch.nn.Conv2d(
            in_channels=mdl.segmentation_head[0].in_channels,
            out_channels=num_classes+1,
            kernel_size=mdl.segmentation_head[0].kernel_size,
            stride=mdl.segmentation_head[0].stride,
            padding=mdl.segmentation_head[0].padding,
        )
    return mdl


def pspnet_b4_mod_family(
    pretrained: bool = False, 
    path: str = "pspnet_b4_mod_family.pt",
    num_classes: int = 6,
):
    """Constructs a PSPNet architecture with an EfficientNet-B4 backbone.
    PSPNet from `"Pyramid Scene Parsing Network" <https://arxiv.org/pdf/1612.01105.pdf>`_.
    EfficientNet from `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/pdf/1905.11946.pdf>`_.
    
    Args:
        pretrained (bool): 
            If True, returns a model pre-trained on WBSig53
        path (str): 
            Path to existing model or where to download checkpoint to
        num_classes (int): 
            Number of output classes; if loading checkpoint and 
            number does not equal 1, final layer will not be loaded from checkpoint
            NOTE: num_classes should equal the total number of classes **without**
            including the background class. That "class" is automatically included.
        
    """
    # Create PSPNet-B4
    mdl = create_pspnet(
        encoder='efficientnet-b4',
        num_classes=6+1,
    )
    if pretrained:
        model_exists = os.path.exists(path)
        if not model_exists:
            file_id = model_urls['pspnet_b4_mod_family']
            dl = gdown.download(id=file_id, output=path)
        mdl.load_state_dict(torch.load(path), strict=False)
    if num_classes != 6:
        mdl.segmentation_head[0] = torch.nn.Conv2d(
            in_channels=mdl.segmentation_head[0].in_channels,
            out_channels=num_classes+1,
            kernel_size=mdl.segmentation_head[0].kernel_size,
            stride=mdl.segmentation_head[0].stride,
            padding=mdl.segmentation_head[0].padding,
        )
    return mdl
