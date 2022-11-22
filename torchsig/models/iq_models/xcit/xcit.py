import os
import timm
import gdown
import torch
from torch import nn


__all__ = ["xcit_nano", "xcit_tiny12"]

file_ids = {
    "xcit_nano": "1c347oGdOd2vQD3vzTqKIv1rxXKfW1Ak6",
    "xcit_tiny12": "1DKd5E3WwEZxt99qCeSIzvgc1AWEEfdue",
}


class ConvDownSampler(nn.Module):
    def __init__(self, in_chans, embed_dim, ds_rate=16):
        super().__init__()
        ds_rate //= 2
        chan = embed_dim // ds_rate
        blocks = [nn.Conv1d(in_chans, chan, 5, 2, 2), nn.BatchNorm1d(chan), nn.SiLU()]

        while ds_rate > 1:
            blocks += [
                nn.Conv1d(chan, 2 * chan, 5, 2, 2),
                nn.BatchNorm1d(2 * chan),
                nn.SiLU(),
            ]
            ds_rate //= 2
            chan = 2 * chan

        blocks += [
            nn.Conv1d(
                chan,
                embed_dim,
                1,
            )
        ]
        self.blocks = nn.Sequential(*blocks)

    def forward(self, X):
        return self.blocks(X)


class Chunker(nn.Module):
    def __init__(self, in_chans, embed_dim, ds_rate=16):
        super().__init__()
        self.embed = nn.Conv1d(in_chans, embed_dim // ds_rate, 7, padding=3)
        self.project = nn.Conv1d((embed_dim // ds_rate) * ds_rate, embed_dim, 1)
        self.ds_rate = ds_rate

    def forward(self, X):
        X = self.embed(X)
        X = torch.cat(
            [
                torch.cat(torch.split(x_i, 1, -1), 1)
                for x_i in torch.split(X, self.ds_rate, -1)
            ],
            -1,
        )
        X = self.project(X)

        return X
    
    
class XCiT(nn.Module):
    def __init__(self, backbone, in_chans=2, ds_rate=2, ds_method="downsample"):
        super().__init__()
        self.backbone = backbone
        W = backbone.num_features
        self.grouper = nn.Conv1d(W, backbone.num_classes, 1)
        if ds_method == "downsample":
            self.backbone.patch_embed = ConvDownSampler(in_chans, W, ds_rate)
        else:
            self.backbone.patch_embed = Chunker(in_chans, W, ds_rate)

    def forward(self, x):
        mdl = self.backbone
        B = x.shape[0]
        x = self.backbone.patch_embed(x)

        Hp, Wp = x.shape[-1], 1
        pos_encoding = (
            mdl.pos_embed(B, Hp, Wp).reshape(B, -1, Hp).permute(0, 2, 1).half()
        )
        x = x.transpose(1, 2) + pos_encoding
        for blk in mdl.blocks:
            x = blk(x, Hp, Wp)
        cls_tokens = mdl.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in mdl.cls_attn_blocks:
            x = blk(x)
        x = mdl.norm(x)
        x = self.grouper(x.transpose(1, 2)[:, :, :1]).squeeze()
        if x.dim() == 2:
            x = x.unsqueeze(0)
        return x
    
    
def xcit_nano(
    pretrained: bool = False, 
    path: str = "xcit_nano.pt",
    num_classes: int = 53,
    drop_path_rate: float = 0.0,
    drop_rate: float = 0.3,
):
    """Constructs a XCiT-Nano architecture from
    `"XCiT: Cross-Covariance Image Transformers" <https://arxiv.org/pdf/2106.09681.pdf>`_.

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
        file_id = file_ids["xcit_nano"]
        dl = gdown.download(id=file_id, output=path)
    mdl = XCiT(
        timm.create_model(
            'xcit_nano_12_p16_224',
            num_classes=53,
            in_chans=2,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
        ),
    )
    if pretrained:
        mdl.load_state_dict(torch.load(path))
    if num_classes!=53:
        mdl.classifier = nn.Linear(mdl.classifier.in_features, num_classes)
    return mdl
    
    
def xcit_tiny12(
    pretrained: bool = False, 
    path: str = "xcit_tiny12.pt",
    num_classes: int = 53,
    drop_path_rate: float = 0.0,
    drop_rate: float = 0.3,
):
    """Constructs a XCiT-Tiny12 architecture from
    `"XCiT: Cross-Covariance Image Transformers" <https://arxiv.org/pdf/2106.09681.pdf>`_.

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
        file_id = file_ids["xcit_tiny12"]
        dl = gdown.download(id=file_id, output=path)
    mdl = XCiT(
        timm.create_model(
            'xcit_tiny_12_p16_224',
            num_classes=53,
            in_chans=2,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
        )
    )
    if pretrained:
        mdl.load_state_dict(torch.load(path))
    if num_classes!=53:
        mdl.classifier = nn.Linear(mdl.classifier.in_features, num_classes)
    return mdl
