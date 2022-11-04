import torch
import numpy as np

from .backbone import EffNetBackbone, ResNet50Backbone
from .pixel_decoder import MSDeformAttnPixelDecoder
from .predictor import MultiScaleMaskedTransformerDecoder


class Mask2FormerModel(torch.nn.Module):
    def __init__(
        self, 
        backbone: torch.nn.Module,
        pixel_decoder: torch.nn.Module,
        predictor: torch.nn.Module,
        num_classes: int = 1,
    ):
        super().__init__()
        self.backbone = backbone
        self.pixel_decoder = pixel_decoder
        self.predictor = predictor
        self.num_classes = num_classes
        
    def forward(self, x):
        # Propagate inputs through model layers
        features = self.backbone(x)
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        predictions = self.predictor(multi_scale_features, mask_features, mask=None)
        return predictions
    
    
def create_backbone(
    backbone: str = 'efficientnet_b0',
) -> torch.nn.Module:
    if 'eff' in backbone:
        if 'b0' in backbone or 'b2' in backbone or 'b4' in backbone:
            network = EffNetBackbone(network=backbone)
        else:
            raise NotImplemented("Only B0, B2, and B4 EffNets are supported at this time")
    elif backbone == 'resnet50':
        network = ResNet50Backbone()
    else:
        raise NotImplemented("Only EfficientNet and ResNet-50 backbones supported at this time.")
    return network
        
    
def create_pixel_decoder(
    pixel_decoder: str = 'multi_scale_deformable_attention',
    backbone: str = 'efficientnet_b0',
    transformer_dropout: float = 0.0,
    transformer_nheads: int = 8,
    transformer_dim_feedforward: int = 2048,
    transformer_enc_layers: int = 0,
    conv_dim: int = 256,
    mask_dim: int = 256,
    norm: str = 'GN',
    common_stride: int = 4,
) -> torch.nn.Module:
    if pixel_decoder == 'multi_scale_deformable_attention':
        network = MSDeformAttnPixelDecoder(
            backbone=backbone,
            transformer_dropout=transformer_dropout,
            transformer_nheads=transformer_nheads,
            transformer_dim_feedforward=transformer_dim_feedforward,
            transformer_enc_layers=transformer_enc_layers,
            conv_dim=conv_dim,
            mask_dim=mask_dim,
            norm=norm,
            common_stride=common_stride,
        )
    else:
        raise NotImplemented("Only multi_scale_deformable_attention supported as a pixel decoder at this time.")
    return network


def create_predictor(
    predictor: str = 'multi_scale_masked_transformer_decoder',
    in_channels: int = 256,
    mask_classification: bool = True,
    num_classes: int = 1,
    hidden_dim: int = 256,
    num_queries: int = 100,
    nheads: int = 8,
    dim_feedforward: int = 2048,
    dec_layers: int = 10,
    pre_norm: bool = False,
    mask_dim: int = 256,
    enforce_input_project: bool = False,
) -> torch.nn.Module:
    if predictor == 'multi_scale_masked_transformer_decoder':
        network = MultiScaleMaskedTransformerDecoder(
            in_channels=in_channels,
            mask_classification=mask_classification,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dec_layers=dec_layers,
            pre_norm=pre_norm,
            mask_dim=mask_dim,
            enforce_input_project=enforce_input_project,
        )
    else:
        raise NotImplemented("Only multi_scale_masked_transformer_decoder supported as predictor at this time.")
    return network
    
    
def create_mask2former(
    backbone: str = 'efficientnet_b0',
    pixel_decoder: str = 'multi_scale_deformable_attention',
    predictor: str = 'multi_scale_masked_transformer_decoder',
    num_classes: int = 1,
) -> torch.nn.Module:
    """
    Function used to build a Mask2Former network
    
    Args:
        TODO
        
    Returns:
        torch.nn.Module
    """
    # Instantiate backbone
    backbone_name = str(backbone)
    backbone = create_backbone(backbone_name)
    
    # Instantiate pixel decoder
    pixel_decoder = create_pixel_decoder(
        pixel_decoder=pixel_decoder,
        backbone=backbone_name,
    )
    
    # Instantiate predictor
    predictor = create_predictor(
        predictor=predictor,
        num_classes=num_classes,
    )
    
    # Create full Mask2Former model
    network = Mask2FormerModel(
        backbone=backbone,
        pixel_decoder=pixel_decoder,
        predictor=predictor,
        num_classes=num_classes
    )
    return network