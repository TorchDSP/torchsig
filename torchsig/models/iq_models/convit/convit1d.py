import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from timm.layers import DropPath, trunc_normal_, Mlp, LayerNorm

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float): Weighting factor for the rare class (default: 1.0).
            gamma (float): Focusing parameter to adjust the rate at which easy examples are down-weighted (default: 2.0).
            reduction (str): Specifies the reduction to apply to the output ('none' | 'mean' | 'sum').
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Predicted logits with shape (batch_size, num_classes).
            targets (Tensor): Ground truth labels with shape (batch_size).
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probabilities of the true classes
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
            
class FocalCosineLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, xent=.1):
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.xent = xent

    def forward(self, inputs, targets):
        cosine_loss = F.cosine_embedding_loss(inputs, F.one_hot(targets, num_classes=inputs.size(-1)), torch.ones(inputs.size(0)).to(inputs.device))
        cent_loss = F.cross_entropy(F.normalize(inputs), targets)
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * cent_loss
        return cosine_loss + self.xent * focal_loss

class PatchEmbed1D(nn.Module):
    """1D Patch Embedding for 1D signals."""
    def __init__(self, seq_length=4096, patch_size=16, in_chans=2, embed_dim=768):
        super().__init__()
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.num_patches = seq_length // patch_size
        self.embed_dim = embed_dim

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.Identity()

    def forward(self, x):
        # x: [B, C, L]
        x = self.proj(x)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        x = self.norm(x)
        return x


class GPSA1D(nn.Module):
    """1D Global Position Self-Attention (GPSA) Layer for 1D sequences."""
    def __init__(
        self,
        dim,
        num_patches,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        locality_strength=1.,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.num_patches = num_patches
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.locality_strength = locality_strength

        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.pos_proj = nn.Linear(2, num_heads)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gating_param = nn.Parameter(torch.ones(self.num_heads))

        # Initialize rel_indices and register it as a buffer
        self.register_buffer('rel_indices', self.get_rel_indices())

    def forward(self, x):
        B, N, C = x.shape
        # No need to move rel_indices here, it's already registered as a buffer
        attn = self.get_attention(x)
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_attention(self, x):
        B, N, C = x.shape
        qk = (
            self.qk(x)
            .reshape(B, N, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k = qk[0], qk[1]
        pos_score = self.rel_indices.expand(B, -1, -1, -1)  # [B, N, N, 2]
        pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)  # [B, num_heads, N, N]
        patch_score = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        patch_score = patch_score.softmax(dim=-1)
        pos_score = pos_score.softmax(dim=-1)

        gating = self.gating_param.view(1, -1, 1, 1)
        attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
        attn = attn / attn.sum(dim=-1, keepdim=True)
        attn = self.attn_drop(attn)
        return attn

    def get_rel_indices(self):
        # Compute relative positions for 1D sequence
        indices = torch.arange(self.num_patches)
        rel_indices = indices.view(-1, 1) - indices.view(1, -1)  # [N, N]
        rel_distances = rel_indices.abs().unsqueeze(-1).float()  # [N, N, 1]
        rel_indices = torch.cat((rel_indices.unsqueeze(-1).float(), rel_distances), dim=-1)  # [N, N, 2]
        rel_indices = rel_indices.unsqueeze(0)  # [1, N, N, 2]
        return rel_indices

    def local_init(self):
        self.v.weight.data.copy_(torch.eye(self.dim))
        self.pos_proj.weight.data *= self.locality_strength


class MHSA1D(nn.Module):
    """1D Multi-Head Self-Attention (MHSA) Layer for 1D sequences."""
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (
            (attn @ v)
            .transpose(1, 2)
            .reshape(B, N, C)
        )
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block1D(nn.Module):
    """Transformer Block with optional GPSA or MHSA attention for 1D sequences."""
    def __init__(
        self,
        dim,
        num_patches,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        proj_drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=LayerNorm,
        use_gpsa=True,
        locality_strength=1.,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.use_gpsa = use_gpsa
        if self.use_gpsa:
            self.attn = GPSA1D(
                dim,
                num_patches=num_patches,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                locality_strength=locality_strength,
            )
        else:
            self.attn = MHSA1D(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConVit1D(nn.Module):
    """Convolutional Vision Transformer (ConViT) for 1D I/Q signal classification."""
    def __init__(
        self,
        seq_length=4096,
        patch_size=16,
        in_chans=2,
        num_classes=10,  # Adjust according to your number of modulation types
        embed_dim=192,
        depth=12,
        num_heads=4,
        mlp_ratio=4.,
        qkv_bias=False,
        drop_rate=0.,
        pos_drop_rate=0.,
        proj_drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=LayerNorm,
        local_up_to_layer=3,
        locality_strength=1.,
        use_pos_embed=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.local_up_to_layer = local_up_to_layer
        self.use_pos_embed = use_pos_embed

        self.patch_embed = PatchEmbed1D(
            seq_length=seq_length,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block1D(
                dim=embed_dim,
                num_patches=num_patches,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                use_gpsa=i < local_up_to_layer,
                locality_strength=locality_strength,
            ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize parameters
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        for n, m in self.named_modules():
            if hasattr(m, 'local_init'):
                m.local_init()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # x: [B, C, L]
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        if self.use_pos_embed:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # [B, 1, embed_dim]
        for u, blk in enumerate(self.blocks):
            if u == self.local_up_to_layer:
                x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, embed_dim]
            x = blk(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        x = x[:, 0]  # Take the class token output
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class ConVit1DLightning(pl.LightningModule):
    """PyTorch Lightning Module for ConVit1D."""
    def __init__(
        self,
        seq_length=4096,
        patch_size=16,
        in_chans=2,
        num_classes=10,
        embed_dim=192,
        depth=12,
        num_heads=4,
        mlp_ratio=4.,
        qkv_bias=False,
        drop_rate=0.,
        pos_drop_rate=0.,
        proj_drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=LayerNorm,
        local_up_to_layer=3,
        locality_strength=1.,
        use_pos_embed=True,
        learning_rate=1e-3,
        alpha=1.0,
        gamma=2.0,
        xent=0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = ConVit1D(
            seq_length=seq_length,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            pos_drop_rate=pos_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            local_up_to_layer=local_up_to_layer,
            locality_strength=locality_strength,
            use_pos_embed=use_pos_embed,
        )
        # self.loss_fn = FocalLoss(alpha=self.hparams.alpha, gamma=self.hparams.gamma)
        # self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = FocalCosineLoss(
            alpha=self.hparams.alpha,
            gamma=self.hparams.gamma,
            xent=self.hparams.xent,
        )        
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.num_classes)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.num_classes)
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: [B, C, L], y: [B]
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return {'val_loss': loss, 'preds': preds, 'targets': y}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    #     return [optimizer], [scheduler]
