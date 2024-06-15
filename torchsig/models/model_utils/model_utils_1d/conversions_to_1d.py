from torchsig.models.model_utils.layer_tools import replace_layers_of_types
from torchsig.models.model_utils.model_utils_1d.layers_1d import GBN1d, SqueezeExcite1d, FastGlobalAvgPool1d

from torch.nn import Conv1d, BatchNorm1d

def conv2d_to_conv1d(layer_2d):
    """
    returns a 1d conv layer corresponding to the input conv2d layer
    no muation is performed
    """
    return Conv1d(layer_2d.in_channels, layer_2d.out_channels, layer_2d.kernel_size[0], 
                           stride=layer_2d.stride[0], padding=layer_2d.padding[0], dilation=layer_2d.dilation[0], 
                           groups=layer_2d.groups, bias=(True if layer_2d.bias != None else False), padding_mode=layer_2d.padding_mode, device=layer_2d.weight.device,
                           dtype=layer_2d.weight.dtype
                          )
    
def batchNorm2d_to_batchNorm1d(layer_2d):
    """
    returns a BatchNorm1d layer corresponding to the input BatchNorm2d layer
    no muation is performed
    """
    return BatchNorm1d(layer_2d.num_features, eps=layer_2d.eps, momentum=layer_2d.momentum,
                           affine=layer_2d.affine, track_running_stats=layer_2d.track_running_stats,
                           device=layer_2d.weight.device, dtype=layer_2d.weight.dtype
                          )

def batchNorm2d_to_GBN1d(layer_2d):
    """
    returns a GBN1d [Ghost Batch Norm] layer corresponding to the input BatchNorm2d layer
    no muation is performed
    """
    return GBN1d(layer_2d.num_features)

def squeezeExcite_to_squeezeExcite1d(layer_2d):
    """
    returns a GBN1d [Ghost Batch Norm] layer corresponding to the input BatchNorm2d layer
    no muation is performed
    """
    return SqueezeExcite1d(layer_2d.conv_reduce.in_channels, reduced_base_chs=layer_2d.conv_reduce.out_channels)

def make_fast_avg_pooling_layer(layer_2d):
    """
    returns a FastGlobalAvgPool1d layer
    """
    return FastGlobalAvgPool1d(flatten=True)

def convert_2d_model_to_1d(model):
    """
    converts a 2d model to a corresponding 1d model by replacing convolutional layers and other 2d layers with their 1d equivalents
    experimental; may not fully convert models with unrecognized layer types
    mutates input model; returns the mutated model
    this function is still under development and may not correctly convert all 2d layer types, or may have unexpected behavior on models that perform reshaping internally
    """
    type_factory_pairs = [
        ('Conv2d', conv2d_to_conv1d),
        ('BatchNorm2d', batchNorm2d_to_GBN1d),
        ('SqueezeExcite', squeezeExcite_to_squeezeExcite1d),
        ('SelectAdaptivePool2d',make_fast_avg_pooling_layer),
    ]
    
    replace_layers_of_types(model, type_factory_pairs)
    return model

