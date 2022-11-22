import timm
import gdown
import torch
import os.path
import numpy as np
from torch import nn

from .utils import non_max_suppression_df, format_preds, format_targets
from .criterion import SetCriterion, HungarianMatcher

    
__all__ = [
    "mask2former_b0", "mask2former_b2", "mask2former_b4",
    "mask2former_b0_mod_family", "mask2former_b2_mod_family", "mask2former_b4_mod_family",
]

model_urls = {
    "mask2former_b0": "1sioOi9k1O3tzxM1Hu5CpME1u9Q3wt_ht",
    "mask2former_b2": "1ZJOSu5jLUS-ZgUmytXdMcyuwHaw5C10b",
    "mask2former_b4": "1xBdw6oGLn7M3JUR7D7p1mbwelcWUsAvj",
    "mask2former_b0_mod_family": "1eRijUw6zuMvPIHNB4-9NwN3rY_1fFA7i",
    "mask2former_b2_mod_family": "1pKAGMALwc3XBg1l14cYDHNFw2ObtHMnx",
    "mask2former_b4_mod_family": "1-_86eGkTDaq9uykgTEZOo1Gky5ITXLJI",
}


def mask2former_b0(
    pretrained: bool = False, 
    path: str = "mask2former_b0.pt",
    num_classes: int = 1,
):
    """Constructs a Mask2Former architecture with an EfficientNet-B0 backbone.
    Mask2Former from `"Masked-attention Mask Transformer for Universal Image Segmentation" <https://arxiv.org/pdf/2112.01527.pdf>`_.
    EfficientNet from `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/pdf/1905.11946.pdf>`_.
    
    Args:
        pretrained (bool): 
            If True, returns a model pre-trained on WBSig53
        path (str): 
            Path to existing model or where to download checkpoint to
        num_classes (int): 
            Number of output classes; if loading checkpoint and 
            number does not equal 1, final layer will not be loaded from checkpoint
        
    """
    from .modules import Mask2FormerModel, create_mask2former
    
    # Create Mask2Former-B0
    mdl = create_mask2former(
        backbone='efficientnet_b0',
        pixel_decoder='multi_scale_deformable_attention',
        predictor='multi_scale_masked_transformer_decoder',
        num_classes=1,
    )
    if pretrained:
        model_exists = os.path.exists(path)
        if not model_exists:
            file_id = model_urls['mask2former_b0']
            dl = gdown.download(id=file_id, output=path)
        mdl.load_state_dict(torch.load(path), strict=False)
    if num_classes != 1:
        raise NotImplemented('Mask2Former implementation does not support finetuning to different class sizes yet.')
    return mdl


def mask2former_b2(
    pretrained: bool = False, 
    path: str = "mask2former_b2.pt",
    num_classes: int = 1,
):
    """Constructs a Mask2Former architecture with an EfficientNet-B2 backbone.
    Mask2Former from `"Masked-attention Mask Transformer for Universal Image Segmentation" <https://arxiv.org/pdf/2112.01527.pdf>`_.
    EfficientNet from `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/pdf/1905.11946.pdf>`_.
    
    Args:
        pretrained (bool): 
            If True, returns a model pre-trained on WBSig53
        path (str): 
            Path to existing model or where to download checkpoint to
        num_classes (int): 
            Number of output classes; if loading checkpoint and 
            number does not equal 1, final layer will not be loaded from checkpoint
        
    """
    from .modules import Mask2FormerModel, create_mask2former
    
    # Create Mask2Former-B2
    mdl = create_mask2former(
        backbone='efficientnet_b2',
        pixel_decoder='multi_scale_deformable_attention',
        predictor='multi_scale_masked_transformer_decoder',
        num_classes=1,
    )
    if pretrained:
        model_exists = os.path.exists(path)
        if not model_exists:
            file_id = model_urls['mask2former_b2']
            dl = gdown.download(id=file_id, output=path)
        mdl.load_state_dict(torch.load(path), strict=False)
    if num_classes != 1:
        raise NotImplemented('Mask2Former implementation does not support finetuning to different class sizes yet.')
    return mdl


def mask2former_b4(
    pretrained: bool = False, 
    path: str = "mask2former_b4.pt",
    num_classes: int = 1,
):
    """Constructs a Mask2Former architecture with an EfficientNet-B4 backbone.
    Mask2Former from `"Masked-attention Mask Transformer for Universal Image Segmentation" <https://arxiv.org/pdf/2112.01527.pdf>`_.
    EfficientNet from `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/pdf/1905.11946.pdf>`_.
    
    Args:
        pretrained (bool): 
            If True, returns a model pre-trained on WBSig53
        path (str): 
            Path to existing model or where to download checkpoint to
        num_classes (int): 
            Number of output classes; if loading checkpoint and 
            number does not equal 1, final layer will not be loaded from checkpoint
        
    """
    from .modules import Mask2FormerModel, create_mask2former
    
    # Create Mask2Former-B4
    mdl = create_mask2former(
        backbone='efficientnet_b4',
        pixel_decoder='multi_scale_deformable_attention',
        predictor='multi_scale_masked_transformer_decoder',
        num_classes=1,
    )
    if pretrained:
        model_exists = os.path.exists(path)
        if not model_exists:
            file_id = model_urls['mask2former_b4']
            dl = gdown.download(id=file_id, output=path)
        mdl.load_state_dict(torch.load(path), strict=False)
    if num_classes != 1:
        raise NotImplemented('Mask2Former implementation does not support finetuning to different class sizes yet.')
    return mdl


def mask2former_b0_mod_family(
    pretrained: bool = False, 
    path: str = "mask2former_b0_mod_family.pt",
    num_classes: int = 6,
):
    """Constructs a Mask2Former architecture with an EfficientNet-B0 backbone.
    Mask2Former from `"Masked-attention Mask Transformer for Universal Image Segmentation" <https://arxiv.org/pdf/2112.01527.pdf>`_.
    EfficientNet from `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/pdf/1905.11946.pdf>`_.
    
    Args:
        pretrained (bool): 
            If True, returns a model pre-trained on WBSig53
        path (str): 
            Path to existing model or where to download checkpoint to
        num_classes (int): 
            Number of output classes; if loading checkpoint and 
            number does not equal 6, final layer will not be loaded from checkpoint
        
    """
    from .modules import Mask2FormerModel, create_mask2former
    
    # Create Mask2Former-B0
    mdl = create_mask2former(
        backbone='efficientnet_b0',
        pixel_decoder='multi_scale_deformable_attention',
        predictor='multi_scale_masked_transformer_decoder',
        num_classes=6,
    )
    if pretrained:
        model_exists = os.path.exists(path)
        if not model_exists:
            file_id = model_urls['mask2former_b0_mod_family']
            dl = gdown.download(id=file_id, output=path)
        mdl.load_state_dict(torch.load(path), strict=False)
    if num_classes != 6:
        raise NotImplemented('Mask2Former implementation does not support finetuning to different class sizes yet.')
    return mdl


def mask2former_b2_mod_family(
    pretrained: bool = False, 
    path: str = "mask2former_b2_mod_family.pt",
    num_classes: int = 6,
):
    """Constructs a Mask2Former architecture with an EfficientNet-B2 backbone.
    Mask2Former from `"Masked-attention Mask Transformer for Universal Image Segmentation" <https://arxiv.org/pdf/2112.01527.pdf>`_.
    EfficientNet from `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/pdf/1905.11946.pdf>`_.
    
    Args:
        pretrained (bool): 
            If True, returns a model pre-trained on WBSig53
        path (str): 
            Path to existing model or where to download checkpoint to
        num_classes (int): 
            Number of output classes; if loading checkpoint and 
            number does not equal 1, final layer will not be loaded from checkpoint
        
    """
    from .modules import Mask2FormerModel, create_mask2former
    
    # Create Mask2Former-B2
    mdl = create_mask2former(
        backbone='efficientnet_b2',
        pixel_decoder='multi_scale_deformable_attention',
        predictor='multi_scale_masked_transformer_decoder',
        num_classes=6,
    )
    if pretrained:
        model_exists = os.path.exists(path)
        if not model_exists:
            file_id = model_urls['mask2former_b2_mod_family']
            dl = gdown.download(id=file_id, output=path)
        mdl.load_state_dict(torch.load(path), strict=False)
    if num_classes != 6:
        raise NotImplemented('Mask2Former implementation does not support finetuning to different class sizes yet.')
    return mdl


def mask2former_b4_mod_family(
    pretrained: bool = False, 
    path: str = "mask2former_b4_mod_family.pt",
    num_classes: int = 6,
):
    """Constructs a Mask2Former architecture with an EfficientNet-B4 backbone.
    Mask2Former from `"Masked-attention Mask Transformer for Universal Image Segmentation" <https://arxiv.org/pdf/2112.01527.pdf>`_.
    EfficientNet from `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/pdf/1905.11946.pdf>`_.
    
    Args:
        pretrained (bool): 
            If True, returns a model pre-trained on WBSig53
        path (str): 
            Path to existing model or where to download checkpoint to
        num_classes (int): 
            Number of output classes; if loading checkpoint and 
            number does not equal 6, final layer will not be loaded from checkpoint
        
    """
    from .modules import Mask2FormerModel, create_mask2former
    
    # Create Mask2Former-B4
    mdl = create_mask2former(
        backbone='efficientnet_b4',
        pixel_decoder='multi_scale_deformable_attention',
        predictor='multi_scale_masked_transformer_decoder',
        num_classes=6,
    )
    if pretrained:
        model_exists = os.path.exists(path)
        if not model_exists:
            file_id = model_urls['mask2former_b4_mod_family']
            dl = gdown.download(id=file_id, output=path)
        mdl.load_state_dict(torch.load(path), strict=False)
    if num_classes != 6:
        raise NotImplemented('Mask2Former implementation does not support finetuning to different class sizes yet.')
    return mdl
