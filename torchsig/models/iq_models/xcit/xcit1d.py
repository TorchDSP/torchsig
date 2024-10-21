import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from torch import Tensor
from typing import Optional, Tuple, List

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["XCiT1d", "XCiTClassifier"]

class XCiT1d(nn.Module):
    """A 1D implementation of the XCiT architecture.

    Args:
        input_channels (int): Number of 1D input channels.
        n_features (int): Number of output features/classes.
        xcit_version (str): Version of XCiT model to use (e.g., 'nano_12_p16_224').
        drop_path_rate (float): Drop path rate for training.
        drop_rate (float): Dropout rate for training.
        ds_method (str): Downsampling method ('downsample' or 'chunk').
        ds_rate (int): Downsampling rate (e.g., 2 for downsampling by a factor of 2).
    """
    def __init__(
        self,
        input_channels: int,
        n_features: int,
        xcit_version: str = "nano_12_p16_224",
        drop_path_rate: float = 0.0,
        drop_rate: float = 0.3,
        ds_method: str = "downsample",
        ds_rate: int = 2
    ):
        super().__init__()

        # Ensure the model name is correct
        model_name = f"xcit_{xcit_version}" if not xcit_version.startswith("xcit_") else xcit_version

        # Create the backbone model
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=n_features,
            in_chans=input_channels,
            drop_path_rate=drop_path_rate,
            drop_rate=drop_rate,
        )

        # Number of features from the backbone
        W = self.backbone.num_features

        # Include the grouper Conv1d layer
        self.grouper = nn.Conv1d(W, n_features, kernel_size=1)

        # Replace the patch embedding with a 1D version
        if ds_method == "downsample":
            self.backbone.patch_embed = ConvDownSampler(input_channels, W, ds_rate)
        elif ds_method == "chunk":
            self.backbone.patch_embed = Chunker(input_channels, W, ds_rate)
        else:
            raise ValueError(
                f"{ds_method} is not a supported downsampling method; currently 'downsample' and 'chunk' are supported"
            )

        # Replace the classifier head with an identity layer (since we use self.grouper)
        self.backbone.head = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        mdl = self.backbone
        B = x.shape[0]

        # Patch embedding
        x = self.backbone.patch_embed(x)  # Shape: [B, C, L]

        # Define H and W for 1D data
        Hp, Wp = x.shape[-1], 1  # Height is sequence length, Width is 1

        # Obtain positional encoding
        pos_encoding = mdl.pos_embed(B, Hp, Wp).reshape(B, -1, Hp).permute(0, 2, 1)

        # Add positional encoding
        x = x.transpose(1, 2) + pos_encoding  # Shape: [B, Hp, C]

        # Apply transformer blocks
        for blk in mdl.blocks:
            x = blk(x, Hp, Wp)

        # Classification token
        cls_tokens = mdl.cls_token.expand(B, -1, -1)  # Shape: [B, 1, C]
        x = torch.cat((cls_tokens, x), dim=1)  # Shape: [B, Hp+1, C]

        # Apply class attention blocks
        for blk in mdl.cls_attn_blocks:
            x = blk(x)

        # Layer normalization
        x = mdl.norm(x)  # Shape: [B, Hp+1, C]

        # Apply the grouper Conv1d to the classification token
        # Extract the classification token (first token)
        cls_token = x[:, 0, :]  # Shape: [B, C]

        # Reshape for Conv1d: [B, C, 1]
        cls_token = cls_token.unsqueeze(-1)  # Shape: [B, C, 1]

        # Apply the grouper Conv1d
        x = self.grouper(cls_token).squeeze(-1)  # Shape: [B, n_features]

        # If x is 1D (batch size 1), ensure it has the correct shape
        if x.dim() == 1:
            x = x.unsqueeze(0)

        return x

class ConvDownSampler(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, ds_rate: int = 16):
        super().__init__()
        # Use a single convolutional layer with appropriate stride
        self.conv = nn.Conv1d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=ds_rate * 2,
            stride=ds_rate,
            padding=ds_rate // 2,
        )
        self.bn = nn.BatchNorm1d(embed_dim)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class Chunker(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, ds_rate: int = 16):
        super().__init__()
        self.ds_rate = ds_rate
        self.embed = nn.Conv1d(in_chans, embed_dim, kernel_size=7, padding=3)
        self.pool = nn.AvgPool1d(kernel_size=ds_rate, stride=ds_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x)  # Shape: [B, embed_dim, L]
        x = self.pool(x)   # Downsample by averaging
        return x

class PositionalEncoding1D(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x: Tensor) -> Tensor:
        B, L, C = x.size()
        position = torch.arange(L, device=x.device).unsqueeze(1)  # Shape: [L, 1]
        div_term = torch.exp(torch.arange(0, C, 2, device=x.device) * (-torch.log(torch.tensor(10000.0)) / C))
        pe = torch.zeros(L, C, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).expand(B, -1, -1)  # Shape: [B, L, C]
        return pe

class XCiTClassifier(LightningModule):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        xcit_version: str = 'tiny_12_p16_224',
        ds_method: str = 'downsample',
        ds_rate: int = 16,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = XCiT1d(
            input_channels=input_channels,
            n_features=num_classes,
            xcit_version=xcit_version,
            ds_method=ds_method,
            ds_rate=ds_rate,
        )
        self.learning_rate = learning_rate
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = FocalLoss(gamma=2.0, alpha=None, reduction='mean')

        # For logging
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx) -> Tensor:
        x, y = batch
        x = x.float() 
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)

        self.train_losses.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        x, y = batch
        x = x.float() 
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.val_losses.append(loss.item())
        self.val_accuracies.append(acc.item())

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    #     return optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [lr_scheduler]

# Metric Tracker for Classifiers
class ClassifierMetrics(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if 'train_loss' in metrics and 'train_acc' in metrics:
            self.train_losses.append(metrics['train_loss'].item())
            self.train_accs.append(metrics['train_acc'].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if 'val_loss' in metrics and 'val_acc' in metrics:
            self.val_losses.append(metrics['val_loss'].item())
            self.val_accs.append(metrics['val_acc'].item())

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Can be a scalar or a tensor of shape [num_classes]
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        ce_loss = F.nll_loss(log_probs, targets, weight=self.alpha, reduction='none', ignore_index=self.ignore_index)
        probs = torch.exp(-ce_loss)
        focal_loss = ((1 - probs) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
