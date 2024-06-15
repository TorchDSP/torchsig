import numpy as np
from torch import sigmoid, cat
from torch.nn import Module, functional, SiLU, Conv1d, BatchNorm1d

class SqueezeExcite1d(Module):
    def __init__(
        self,
        in_chs,
        se_ratio=0.25,
        reduced_base_chs=None,
        act_layer=SiLU,
        gate_fn=sigmoid,
        divisor=1,
        **_,
    ):
        super(SqueezeExcite1d, self).__init__()
        reduced_chs = reduced_base_chs
        self.conv_reduce = Conv1d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = Conv1d(reduced_chs, in_chs, 1, bias=True)
        self.gate_fn = gate_fn

    def forward(self, x):
        x_se = x.mean((2,), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate_fn(x_se)


class FastGlobalAvgPool1d(Module):
    def __init__(self, flatten=False):
        super(FastGlobalAvgPool1d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1)


class GBN1d(Module):
    """
    Ghost Batch Normalization
    https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.1):
        super(GBN1d, self).__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]

        return cat(res, dim=0)

class ImageFrom1D(Module):
    """
    A layer for reshaping (batch_size x n_channels x N) 1d signal data to (batch_size x n_channels x new_y x new_x)
    where new_y and new_x are N**0.5 padded to as near a perfect square as can be formed from N points without adding more than a full row of padding
    """
    def __init__(self, n_channels = 3):
        super().__init__()
        self.n_channels = n_channels

    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size,self.n_channels, -1)
        n_values = x_flat.size(2)
        y_dim = np.sqrt(n_values).astype(np.int32)
        x_dim = (n_values // y_dim)
        n_pad = ((batch_size * self.n_channels * y_dim * x_dim) - (x_flat.size(0) * x_flat.size(1) * x_flat.size(2)))//(x_flat.size(0) * x_flat.size(1))
        x_flat = functional.pad(x_flat, [0,n_pad])
        img_tensor = x_flat.reshape(batch_size,self.n_channels,y_dim,x_dim)
        return img_tensor

