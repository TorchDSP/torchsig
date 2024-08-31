import os.path

import gdown
import numpy as np
import timm
import torch
from torch import nn

__all__ = ["efficientnet_b0", "efficientnet_b2", "efficientnet_b4"]

file_ids = {
    "efficientnet_b0": "1ZQIBRZJiwwjeP4rB7HxxFzFro7RbxihG",
    "efficientnet_b2": "1yaPZS5bbf6npHfUVdswvUnsJb8rDHlaa",
    "efficientnet_b4": "1KCoLY5X0rIc_6ArmZRdkxZOOusIHN6in",
}


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_chs,
        se_ratio=0.25,
        reduced_base_chs=None,
        act_layer=nn.SiLU,
        gate_fn=torch.sigmoid,
        divisor=1,
        **_,
    ):
        super(SqueezeExcite, self).__init__()
        reduced_chs = reduced_base_chs
        self.conv_reduce = nn.Conv1d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv1d(reduced_chs, in_chs, 1, bias=True)
        self.gate_fn = gate_fn

    def forward(self, x):
        x_se = x.mean((2,), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate_fn(x_se)


class FastGlobalAvgPool1d(nn.Module):
    def __init__(self, flatten=False):
        super(FastGlobalAvgPool1d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1)


class GBN(torch.nn.Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.1):
        super(GBN, self).__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = nn.BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]

        return torch.cat(res, dim=0)


def replace_bn(parent):
    for n, m in parent.named_children():
        if type(m) is nn.BatchNorm2d:
            setattr(
                parent,
                n,
                GBN(m.num_features),
            )
        else:
            replace_bn(m)


def replace_se(parent):
    for n, m in parent.named_children():
        if type(m) is timm.models._efficientnet_blocks.SqueezeExcite:
            setattr(
                parent,
                n,
                SqueezeExcite(
                    m.conv_reduce.in_channels,
                    reduced_base_chs=m.conv_reduce.out_channels,
                ),
            )
        else:
            replace_se(m)


def replace_conv_effnet(parent, ds_rate):
    for n, m in parent.named_children():
        if type(m) is nn.Conv2d:
            if ds_rate == 2:
                setattr(
                    parent,
                    n,
                    nn.Conv1d(
                        m.in_channels,
                        m.out_channels,
                        kernel_size=m.kernel_size[0],
                        stride=m.stride[0],
                        padding=m.padding[0],
                        bias=m.kernel_size[0],
                        groups=m.groups,
                    ),
                )
            else:
                setattr(
                    parent,
                    n,
                    nn.Conv1d(
                        m.in_channels,
                        m.out_channels,
                        kernel_size=m.kernel_size[0] if m.kernel_size[0] == 1 else 5,
                        stride=m.stride[0] if m.stride[0] == 1 else ds_rate,
                        padding=m.padding[0] if m.padding[0] == 0 else 2,
                        bias=m.kernel_size[0],
                        groups=m.groups,
                    ),
                )
        else:
            replace_conv_effnet(m, ds_rate)


def create_effnet(network, ds_rate=2):
    replace_se(network)
    replace_bn(network)
    replace_conv_effnet(network, ds_rate)
    network.global_pool = FastGlobalAvgPool1d(flatten=True)
    return network


def efficientnet_b0(
    pretrained: bool = False,
    path: str = "efficientnet_b0.pt",
    num_classes: int = 53,
    drop_path_rate: float = 0.2,
    drop_rate: float = 0.3,
):
    """Constructs a EfficientNet B0 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool):
            If True, returns a model pre-trained on Sig53

        path (str):
            Path to existing model or where to download checkpoint to

        num_classes (int):
            Number of output classes; if loading checkpoint and number does not
            equal 53, final layer will not be loaded from checkpoint

        drop_path_rate (float):
            Drop path rate for training

        drop_rate (float):
            Dropout rate for training

    """
    model_exists = os.path.exists(path)
    if not model_exists and pretrained:
        file_id = file_ids["efficientnet_b0"]
        dl = gdown.download(id=file_id, output=path)
    mdl = create_effnet(
        timm.create_model(
            "efficientnet_b0",
            num_classes=53,
            in_chans=2,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
        )
    )
    if pretrained:
        mdl.load_state_dict(torch.load(path))
    if num_classes != 53:
        mdl.classifier = nn.Linear(mdl.classifier.in_features, num_classes)
    return mdl


def efficientnet_b2(
    pretrained: bool = False,
    path: str = "efficientnet_b2.pt",
    num_classes: int = 53,
    drop_path_rate: float = 0.2,
    drop_rate: float = 0.3,
):
    """Constructs a EfficientNet B2 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool):
            If True, returns a model pre-trained on Sig53

        path (str):
            Path to existing model or where to download checkpoint to

        num_classes (int):
            Number of output classes; if loading checkpoint and number does not
            equal 53, final layer will not be loaded from checkpoint

        drop_path_rate (float):
            Drop path rate for training

        drop_rate (float):
            Dropout rate for training

    """
    model_exists = os.path.exists(path)
    if not model_exists and pretrained:
        file_id = file_ids["efficientnet_b2"]
        dl = gdown.download(id=file_id, output=path)
    mdl = create_effnet(
        timm.create_model(
            "efficientnet_b2",
            num_classes=53,
            in_chans=2,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
        )
    )
    if pretrained:
        mdl.load_state_dict(torch.load(path))
    if num_classes != 53:
        mdl.classifier = nn.Linear(mdl.classifier.in_features, num_classes)
    return mdl


def efficientnet_b4(
    pretrained: bool = False,
    path: str = "efficientnet_b4.pt",
    num_classes: int = 53,
    drop_path_rate: float = 0.2,
    drop_rate: float = 0.3,
):
    """Constructs a EfficientNet B4 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool):
            If True, returns a model pre-trained on Sig53

        path (str):
            Path to existing model or where to download checkpoint to

        num_classes (int):
            Number of output classes; if loading checkpoint and number does not
            equal 53, final layer will not be loaded from checkpoint

        drop_path_rate (float):
            Drop path rate for training

        drop_rate (float):
            Dropout rate for training

    """
    model_exists = os.path.exists(path)
    if not model_exists and pretrained:
        file_id = file_ids["efficientnet_b4"]
        dl = gdown.download(id=file_id, output=path)
    mdl = create_effnet(
        timm.create_model(
            "efficientnet_b4",
            num_classes=53,
            in_chans=2,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
        )
    )
    if pretrained:
        mdl.load_state_dict(torch.load(path))
    if num_classes != 53:
        mdl.classifier = nn.Linear(mdl.classifier.in_features, num_classes)
    return mdl
