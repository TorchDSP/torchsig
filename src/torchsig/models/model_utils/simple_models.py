#A script for defining some useful functions and classes for building models

import numpy as np

from torch.nn import ELU, Sequential, Conv2d, Conv1d, ConvTranspose2d, BatchNorm1d, BatchNorm2d, Linear

STANDARD_ACTIVATION_FUNCTION = ELU

def convnet_block_2d(in_width, out_width, kernel_shape = [5,5], activation_fn = None):
    """
    returns a block of layers consisting of a 2d convolution, batch normalization, and activation function,
    with the input and output channels given by in_width and out_with, a kernel given by kernel_shape, and using the given activation_fn
    if no activation function is provided, this defaults to ELU
    """
    if not activation_fn:
        activation_fn = STANDARD_ACTIVATION_FUNCTION
    return Sequential(
        Conv2d(in_width, out_width, kernel_shape, padding=[kernel_dim//2 for kernel_dim in kernel_shape]),
        BatchNorm2d(out_width),
        activation_fn()
    )

def convnet_block_1d(in_width, out_width, kernel_dim = 5, activation_fn = None):
    """
    1d version of convnet_block_2d above
    """
    if not activation_fn:
        activation_fn = STANDARD_ACTIVATION_FUNCTION
    return Sequential(
        Conv1d(in_width, out_width, kernel_dim, padding=kernel_dim//2),
        BatchNorm1d(out_width),
        activation_fn()
    )

def dense_block(in_width, out_width, activation_fn = None):
    """
    returns a block of layers consisting of a 2d convolution, batch normalization, and activation function,
    with the input and output channels given by in_width and out_with, a kernel given by kernel_shape, and using the given activation_fn
    if no activation function is provided, this defaults to ELU
    """
    if not activation_fn:
        activation_fn = STANDARD_ACTIVATION_FUNCTION
    return Sequential(
        Linear(in_width, out_width),
        BatchNorm1d(out_width),
        activation_fn()
    )

def simple_convnet_2d(layer_width_list):
    """
    takes in a list or tuple of convoluional channel widths and returns a sequential model with those widths
    used to quickly prototype convolutional neural nets;
    for example, simple_convnet_2d([3,8,32,64,64,1]) would return a model with 5 convolutional layers that takes in 
    an X by Y image with 3 color channels and outputs an X by Y image with a single channel.
    Because the returned model doesnt include pooling, striding, or dilation, etc., it does not reduce the scale of the input except possibly in the channel dimension
    As such, it can take up a great deal of memory, and should not be used by itself to perform complicated tasks on large images
    """
    layers = []
    prev_width = layer_width_list[0]
    for layer_width in layer_width_list[1:]:
        layers += [convnet_block_2d(prev_width, layer_width)]
        prev_width = layer_width
    layers += [Conv2d(prev_width, prev_width, [1,1])]
    return Sequential(*layers)

def simple_convnet_1d(layer_width_list):
    """
    1d version of simple_convnet_2d above
    """
    layers = []
    prev_width = layer_width_list[0]
    for layer_width in layer_width_list[1:]:
        layers += [convnet_block_1d(prev_width, layer_width)]
        prev_width = layer_width
    layers += [Conv1d(prev_width, prev_width, 1)]
    return Sequential(*layers)

def simple_densenet(layer_width_list):
    """
    takes in a list or tuple of dense layer widths and returns a sequential model with those widths
    used to quickly prototype simple feed-forward neural nets;
    for example, simple_densenet([6,8,32,64,64,1]) would return a model with fully-connected linear layers that takes in 
    a vecor of length 6 and outputs a single value.
    """
    layers = []
    prev_width = layer_width_list[0]
    for layer_width in layer_width_list[1:]:
        layers += [dense_block(prev_width, layer_width)]
        prev_width = layer_width
    layers += [Linear(prev_width, prev_width)]
    return Sequential(*layers)

def double_image_scale_2d(width = 1, kernel_shape = [5,5], activation_fn = None):
    """
    doubles the scale dimensions of an image of channel width width using a transposd convolution of kernel shape kernel_shape
    calls batch norm and an activation function provided by activation_fn on the result
    if no function is provided, this defaults to ELU
    """
    if not activation_fn:
        activation_fn = ELU
    return Sequential(
        ConvTranspose2d(width, width, kernel_shape, padding=[kernel_dim//2 for kernel_dim in kernel_shape], stride=2, output_padding=1),
        BatchNorm2d(width),
        activation_fn()
    )