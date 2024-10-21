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

__all__ = ["InceptionTime"]

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
            
class InceptionModule(nn.Module):
    def __init__(self, in_channels, num_filters=32, kernel_sizes=[9, 19, 39], bottleneck_channels=32, activation=nn.ReLU()):
        super(InceptionModule, self).__init__()

        # Determine effective number of input channels after bottleneck
        if in_channels > bottleneck_channels:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            effective_in_channels = bottleneck_channels
        else:
            self.bottleneck = nn.Identity()
            effective_in_channels = in_channels  # Use the actual input channels

        # Convolutional branches with different kernel sizes
        self.conv1 = nn.Conv1d(effective_in_channels, num_filters, kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2, bias=False)
        self.conv2 = nn.Conv1d(effective_in_channels, num_filters, kernel_size=kernel_sizes[1], padding=kernel_sizes[1] // 2, bias=False)
        self.conv3 = nn.Conv1d(effective_in_channels, num_filters, kernel_size=kernel_sizes[2], padding=kernel_sizes[2] // 2, bias=False)

        # Max pooling branch
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(in_channels, num_filters, kernel_size=1, bias=False)

        # Batch normalization and activation
        self.bn = nn.BatchNorm1d(num_filters * 4)
        self.activation = activation

    def forward(self, x):
        input_res = x  # For the residual connection

        x = self.bottleneck(x)

        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        pool = self.maxpool(input_res)
        conv4 = self.conv4(pool)

        # Concatenate all convolutional outputs
        x = torch.cat([conv1, conv2, conv3, conv4], dim=1)
        x = self.bn(x)
        x = self.activation(x)
        return x

class InceptionTime(LightningModule):
    def __init__(self, num_classes, input_channels=2, num_modules=6, learning_rate=1e-3):
        super(InceptionTime, self).__init__()
        self.save_hyperparameters()
        num_filters = 32
        self.learning_rate = learning_rate
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = FocalLoss(gamma=2.0, alpha=None, reduction='mean')


        # Stack multiple Inception modules
        self.inception_blocks = nn.ModuleList()
        for i in range(num_modules):
            in_ch = input_channels if i == 0 else num_filters * 4
            self.inception_blocks.append(InceptionModule(in_ch, num_filters=num_filters))

        # Global Average Pooling and Fully Connected layer
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters * 4, num_classes)

    def forward(self, x):
        for block in self.inception_blocks:
            x = block(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_acc', acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    #     return optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [lr_scheduler]
