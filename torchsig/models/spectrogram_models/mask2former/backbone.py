import timm
import numpy as np
import torch
import torch.nn as nn


class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = timm.create_model('resnet50', in_chans=2).float()
        
    def forward(self, x):
        features = {}
        layers = list(self.resnet50.children())
        for i, layer in enumerate(layers):
            x = layer(x)
            if isinstance(layer, nn.Sequential):
                features[str(len(features))] = x
        return features     
    
    
class EffNetBackbone(nn.Module):
    def __init__(self, network='efficientnet_b0'):
        super().__init__()
        self.network = timm.create_model(network, in_chans=2).float()
        
    def forward(self, x):
        features = {}
        layers = list(self.network.children())
        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Sequential):
                for ii, blocks in enumerate(layer):
                    x = blocks(x)                 
                    if isinstance(blocks, nn.Sequential):
                        features[str(len(features))] = x
            else:
                x = layer(x)
        return features