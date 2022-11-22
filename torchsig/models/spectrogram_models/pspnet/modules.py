import numpy as np
import torch
from torch import nn
import segmentation_models_pytorch as smp

from .utils import replace_bn


class BootstrappedCrossEntropy(nn.Module):
    def __init__(
        self, K: float = 0.15, criterion: nn.Module = None, momentum: float = 0.99998,
    ):
        super(BootstrappedCrossEntropy, self).__init__()
        assert criterion != None, "you must give a criterion function"
        self.criterion = criterion
        self.K = K
        self.momentum = momentum

    def forward(self, pred, target, step):
        B, C, H, W = pred.shape
        num = int(self.K * B * H * W * max((self.momentum ** step), self.K))
        loss = self.criterion(pred, target)
        loss = loss.view(-1)
        tk = torch.argsort(loss, descending=True)
        TK = loss[tk[num - 1]]
        loss = loss[loss >= TK]
        return loss.mean()
            
            
def create_pspnet(
    encoder: str = 'efficientnet-b0',
    num_classes: int = 53,
) -> torch.nn.Module:
    """
    Function used to build a PSPNet network
    
    Args:
        TODO
        
    Returns:
        torch.nn.Module
    """
    # Create PSPNet using the SMP library
    # Note that the encoder is instantiated within the PSPNet call
    network = smp.PSPNet(
        encoder_name=encoder,
        in_channels=2,
        classes=num_classes,
    )
    
    # Replace batch norm with group norm
    replace_bn(network)
    
    return network