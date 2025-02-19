from typing import List

import timm
import torch
from scipy import interpolate
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss

from .criterion import dice_loss, nested_tensor_from_tensor_list
from .utils import (
    accuracy,
    box_cxcywh_to_xyxy,
    drop_classifier,
    find_output_features,
    generalized_box_iou,
    get_world_size,
    is_dist_avail_and_initialized,
    xcit_name_to_timm_name,
)

class ConvDownSampler(torch.nn.Module):
    def __init__(self, in_chans, embed_dim, ds_rate=16):
        super().__init__()
        ds_rate //= 2
        chan = embed_dim // ds_rate
        blocks = [
            torch.nn.Conv2d(in_chans, chan, (5,5), 2, 2), 
            torch.nn.BatchNorm2d(chan), 
            torch.nn.SiLU()
        ]

        while ds_rate > 1:
            blocks += [
                torch.nn.Conv2d(chan, 2 * chan, (5,5), 2, 2),
                torch.nn.BatchNorm2d(2 * chan),
                torch.nn.SiLU(),
            ]
            ds_rate //= 2
            chan = 2 * chan

        blocks += [
            torch.nn.Conv2d(
                chan,
                embed_dim,
                (1,1),
            )
        ]
        self.blocks = torch.nn.Sequential(*blocks)

    def forward(self, X):
        return self.blocks(X)


class Chunker(torch.nn.Module):
    def __init__(self, in_chans, embed_dim, ds_rate=16):
        super().__init__()
        self.embed = torch.nn.Conv2d(in_chans, embed_dim // ds_rate, (7,7), padding=3)
        self.project = torch.nn.Conv2d((embed_dim // ds_rate) * ds_rate, embed_dim, (1,1))
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

class XCiT(torch.nn.Module):
    def __init__(self, backbone, in_chans=2, num_objects=50, ds_rate=2, ds_method="downsample"):
        super().__init__()
        self.backbone = backbone
        self.num_objects = num_objects
        W = backbone.num_features
        self.grouper = torch.nn.Conv1d(W, backbone.num_classes, 1)
        if ds_method == "downsample":
            self.backbone.patch_embed = ConvDownSampler(in_chans, W, ds_rate)
        else:
            self.backbone.patch_embed = Chunker(in_chans, W, ds_rate)

    def forward(self, x):
        mdl = self.backbone
        B = x.shape[0]
        x = self.backbone.patch_embed(x)

        Hp, Wp = x.shape[-2], x.shape[-1]
        pos_encoding = (
            mdl.pos_embed(B, Hp, Wp).reshape(B, -1, Hp*Wp).permute(0, 2, 1).half()
        )
        x = x.reshape(B, -1, Hp*Wp).permute(0, 2,1) + pos_encoding
        for blk in mdl.blocks:
            x = blk(x, Hp, Wp)
        cls_tokens = mdl.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in mdl.cls_attn_blocks:
            x = blk(x)
        x = mdl.norm(x)
        x = self.grouper(x.transpose(1, 2)[:, :, :self.num_objects])
        x = x.squeeze()
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = x.transpose(1,2)
        return x


class MLP(torch.nn.Module):
    """Very simple multi-layer perceptron (also called FFN) from DETR repo"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETRModel(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        transformer: torch.nn.Module,
        num_classes: int = 61,
        num_objects: int = 50,
        hidden_dim: int = 256,
    ):
        super().__init__()
        # Convolutional backbone
        self.backbone = backbone

        # Conversion layer
        self.conv = torch.nn.Conv2d(
            in_channels=find_output_features(self.backbone),
            out_channels=hidden_dim,
            kernel_size=1,
        )

        # Transformer
        self.transformer = transformer

        # Prediction heads, one extra class for predicting non-empty slots
        self.linear_class = torch.nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, x):
        # Propagate inputs through backbone
        x = self.backbone(x)

        # Convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # Propagate through the transformer
        h = self.transformer(h)

        # Project transformer outputs to class labels and bounding boxes
        return {
            "pred_logits": self.linear_class(h),
            "pred_boxes": self.linear_bbox(h).sigmoid(),
        }


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes: int = 1,
        class_loss_coef: float = 1.0,
        bbox_loss_coef: float = 5.0,
        giou_loss_coef: float = 2.0,
        eos_coef: float = 0.1,
        losses: List[str] = ["labels", "boxes", "cardinality"],
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = {
            "loss_ce": class_loss_coef,
            "loss_bbox": bbox_loss_coef,
            "loss_giou": giou_loss_coef,
        }
        self.matcher = HungarianMatcher(
            cost_class=self.weight_dict["loss_ce"],
            cost_bbox=self.weight_dict["loss_bbox"],
            cost_giou=self.weight_dict["loss_giou"],
        )
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {"loss_ce": loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(
            src_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(
        self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1
    ):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = (
            outputs["pred_logits"].flatten(0, 1).softmax(-1)
        )  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        )
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]


def create_detr(
    backbone: str = "efficientnet_b0",
    transformer: str = "xcit-nano",
    num_classes: int = 61,
    num_objects: int = 50,
    hidden_dim: int = 256,
    drop_rate_backbone: float = 0.2,
    drop_path_rate_backbone: float = 0.2,
    drop_path_rate_transformer: float = 0.1,
    ds_rate_transformer: int = 2,
    ds_method_transformer: str = "chunker",
) -> torch.nn.Module:
    """
    Function used to build a DETR network

    Args:
        TODO

    Returns:
        torch.nn.Module

    """
    # build backbone
    if "eff" in backbone:
        backbone_arch = timm.create_model(
            model_name=backbone,
            in_chans=2,
            drop_rate=drop_rate_backbone,
            drop_path_rate=drop_path_rate_backbone,
        )
        backbone_arch = drop_classifier(backbone_arch)
    else:
        raise NotImplementedError(
            "Only EfficientNet backbones are supported right now."
        )

    # Build transformer
    if "xcit" in transformer:
        # map short name to timm name
        model_name = xcit_name_to_timm_name(transformer)

        # build transformer
        transformer_arch = XCiT(
            backbone=timm.create_model(
                model_name=model_name,
                drop_path_rate=drop_path_rate_transformer,
                in_chans=hidden_dim,
                num_classes=hidden_dim,
            ),
            in_chans=hidden_dim,
            num_objects=num_objects,
            ds_rate=ds_rate_transformer,
            ds_method=ds_method_transformer,
        )

    else:
        raise NotImplementedError("Only XCiT transformers are supported right now.")

    # Build full DETR network
    network = DETRModel(
        backbone_arch,
        transformer_arch,
        num_classes=num_classes,
        num_objects=num_objects,
        hidden_dim=hidden_dim,
    )

    return network
