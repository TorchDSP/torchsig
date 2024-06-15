import timm
from torch import cat
from torch.nn import Module, Conv1d, Linear

from torchsig.models.model_utils.model_utils_1d.iq_sampling import ConvDownSampler, Chunker

__all__ = ["XCiT1d"]

class XCiT1d(Module):
    """A 1d implementation of the XCiT architecture from
    `"XCiT: Cross-Covariance Image Transformers" <https://arxiv.org/pdf/2106.09681.pdf>`_.

    Args:

        input_channels (int):
            Number of 1d input channels; e.g., common practice is to split complex number time-series data into 2 channels, representing the real and imaginary parts respectively
        
        n_features (int):
            Number of output features; should be the number of classes when used directly for classification

        xcit_version (str):
            Specifies the version of efficientnet to use. See the timm xcit documentation for details. Examples are 'nano_12_p16_224', and 'xcit_tiny_12_p16_224'

        drop_path_rate (float):
            Drop path rate for training

        drop_rate (float):
            Dropout rate for training
        
        ds_method (str):
            Specifies the downsampling method to use in the model. Currently convolutional downsampling and chunking are supported, using string arguments 'downsample' and 'chunk' respectively

        ds_rate (int):
            Specifies the downsampling rate; e.g., ds_rate=2 will downsample the imput by a factor of 2
    """
    def __init__(self, 
        input_channels: int,
        n_features: int,
        xcit_version: str = "nano_12_p16_224",
        drop_path_rate: float = 0.0,
        drop_rate: float = 0.3,
        ds_method: str = "downsample",
        ds_rate: int = 2):
        
        super().__init__()
        self.backbone = timm.create_model(
            "xcit_" + xcit_version,
            num_classes=n_features,
            in_chans=input_channels,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
        )
        
        W = self.backbone.num_features
        self.grouper = Conv1d(W, n_features, 1)
        if ds_method == "downsample":
            self.backbone.patch_embed = ConvDownSampler(input_channels, W, ds_rate)
        elif ds_method == "chunk":
            self.backbone.patch_embed = Chunker(input_channels, W, ds_rate)
        else:
            raise ValueError(ds_method + " is not a supported downsampling method; currently 'downsample', and 'chunk' are supported")

        self.backbone.head = Linear(self.backbone.head.in_features, n_features)

    def forward(self, x):
        mdl = self.backbone
        B = x.shape[0]
        x = self.backbone.patch_embed(x)

        Hp, Wp = x.shape[-1], 1
        pos_encoding = mdl.pos_embed(B, Hp, Wp).reshape(B, -1, Hp).permute(0, 2, 1).half()
        x = x.transpose(1, 2) + pos_encoding
        for blk in mdl.blocks:
            x = blk(x, Hp, Wp)
        cls_tokens = mdl.cls_token.expand(B, -1, -1)
        x = cat((cls_tokens, x), dim=1)
        for blk in mdl.cls_attn_blocks:
            x = blk(x)
        x = mdl.norm(x)
        x = self.grouper(x.transpose(1, 2)[:, :, :1]).squeeze()
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return x
