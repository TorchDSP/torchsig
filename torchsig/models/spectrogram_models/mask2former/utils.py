import math
import numpy as np
import sympy
import timm
import torch
from torch import nn
from torch import Tensor
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
import torchvision
from torchvision.ops.boxes import box_area
from torchvision.ops import masks_to_boxes
from typing import List, Optional


def drop_classifier(parent):
    return torch.nn.Sequential(*list(parent.children())[:-2])


def find_output_features(parent, num_features=0):
    for n, m in parent.named_children():
        if type(m) is torch.nn.Conv2d:
            num_features = m.out_channels
        else:
            num_features = find_output_features(m, num_features)
    return num_features


# Several functions below pulled from public DETR repo: https://github.com/facebookresearch/detr
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def collate_fn(batch):
    return tuple(zip(*batch))
    

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def calc_area(box):
    return max(0,(box[1] - box[0])) * max(0,(box[3] - box[2]))


def calc_iou(box1, box2):
    area1 = calc_area(box1)
    area2 = calc_area(box2)
    inter_x1 = max(box1[0], box2[0])
    inter_x2 = min(box1[1], box2[1])
    inter_y1 = max(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter_area = max(0,calc_area([inter_x1, inter_x2, inter_y1, inter_y2]))
    union = area1 + area2 - inter_area
    iou = inter_area / union
    return iou


def non_max_suppression_df(detected_signals_df, iou_threshold=0.75):
    valid_indices = list(detected_signals_df.index)
    remove_indices = []
    for det_idx in valid_indices:
        for det_jdx in valid_indices:
            if det_idx >= det_jdx:
                continue
                
            # Check if same class
            sig1_class = detected_signals_df.loc[det_idx]['Class']
            sig2_class = detected_signals_df.loc[det_jdx]['Class']

            if sig1_class != sig2_class:
                continue

            #  convert df to box lists: (x1,x2,y1,y2)
            sig1 = [
                detected_signals_df.loc[det_idx]['CenterTimePixel']-detected_signals_df.loc[det_idx]['DurationPixel']/2,
                detected_signals_df.loc[det_idx]['CenterTimePixel']+detected_signals_df.loc[det_idx]['DurationPixel']/2,
                detected_signals_df.loc[det_idx]['CenterFreqPixel']-detected_signals_df.loc[det_idx]['BandwidthPixel']/2,
                detected_signals_df.loc[det_idx]['CenterFreqPixel']+detected_signals_df.loc[det_idx]['BandwidthPixel']/2
            ]
            sig2 = [
                detected_signals_df.loc[det_jdx]['CenterTimePixel']-detected_signals_df.loc[det_jdx]['DurationPixel']/2,
                detected_signals_df.loc[det_jdx]['CenterTimePixel']+detected_signals_df.loc[det_jdx]['DurationPixel']/2,
                detected_signals_df.loc[det_jdx]['CenterFreqPixel']-detected_signals_df.loc[det_jdx]['BandwidthPixel']/2,
                detected_signals_df.loc[det_jdx]['CenterFreqPixel']+detected_signals_df.loc[det_jdx]['BandwidthPixel']/2
            ]

            iou_score = calc_iou(sig1, sig2)

            if iou_score > iou_threshold:
                # Probably the same signal, take higher confidence signal
                sig1_prob = detected_signals_df.loc[det_idx]['Probability']
                sig2_prob = detected_signals_df.loc[det_jdx]['Probability']
                dup_idx = det_idx if sig1_prob < sig2_prob else det_jdx
                
                # remove from valid_indices
                if dup_idx in valid_indices and dup_idx not in remove_indices:
                    remove_indices.append(dup_idx) 

    remove_indices = sorted(remove_indices)
    for idx in range(len(remove_indices)-1,-1,-1):
        valid_indices.remove(remove_indices[idx])
        
    detected_signals_df = detected_signals_df.loc[valid_indices].reset_index(drop=True)
    detected_signals_df['DetectionIdx'] = detected_signals_df.index
    return detected_signals_df


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_wait_steps,
    num_training_steps,
    num_cycles=0.5,
    last_epoch=-1,
):
    def lr_lambda(current_step):
        if current_step < num_wait_steps:
            return 0.0
        if current_step < num_warmup_steps + num_wait_steps:
            return float(current_step - num_wait_steps) / max(
                1, float(num_warmup_steps)
            )
        progress = float(current_step - (num_warmup_steps + num_wait_steps)) / float(
            max(1, num_training_steps - (num_warmup_steps + num_wait_steps))
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def add_weight_decay(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bn" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay}]


def format_preds(preds):
    map_preds = []
    for (i, (det_logits, det_masks)) in enumerate(zip(preds['pred_logits'], preds['pred_masks'])):
        boxes = []
        scores = []
        labels = []

        # Convert Mask2Former output format to expected bboxes
        num_objs = 0
        pred = {}
        pred['pred_logits'] = det_logits
        pred['pred_masks'] = det_masks
        
        det_list = []
        for obj_idx in range(pred['pred_logits'].shape[0]):
            probs = pred['pred_logits'][obj_idx].softmax(-1)
            max_prob = probs.max().cpu().detach().numpy()
            max_class = probs.argmax().cpu().detach().numpy()
            if max_class != (pred['pred_logits'].shape[1] - 1) and max_prob >= 0.5:
                mask = torch.sigmoid(pred['pred_masks'][obj_idx])
                mask[mask > 0.5] = 1.0
                mask[mask != 1.0] = 0.0
                if mask.sum() > 0.0:
                    x1y1x2y2 = masks_to_boxes(mask.unsqueeze(0)).cpu().numpy()[0]
                    x1y1x2y2 = x1y1x2y2 / (pred['pred_masks'].shape[-1]-1) * 511 # Upscale
                    x1 = x1y1x2y2[0]
                    y1 = x1y1x2y2[1]
                    x2 = x1y1x2y2[2]
                    y2 = x1y1x2y2[3]

                    boxes.append([x1, y1, x2, y2])
                    scores.extend([float(max_prob)])
                    labels.extend([int(max_class)])

        curr_pred = dict(
            boxes=torch.tensor(boxes).to("cuda"),
            scores=torch.tensor(scores).to("cuda"),
            labels=torch.IntTensor(labels).to("cuda"),
        )
        
        map_preds.append(curr_pred)
            
    return map_preds


def format_targets(labels):
    map_targets = []
        
    for i, label in enumerate(labels):
        boxes = []
        scores = []
        labels = []
    
        for label_obj_idx in range(len(label['labels'])):
            mask = label['masks'][label_obj_idx]
            if mask.sum() > 0.0:    
                x1y1x2y2 = masks_to_boxes(mask.unsqueeze(0)).numpy()[0]
                x1 = x1y1x2y2[0]
                y1 = x1y1x2y2[1]
                x2 = x1y1x2y2[2]
                y2 = x1y1x2y2[3]

                boxes.append([x1, y1, x2, y2])
                labels.extend([int(label['labels'][label_obj_idx])])
            
        curr_target = dict(
            boxes=torch.tensor(boxes).to("cuda"),
            labels=torch.IntTensor(labels).to("cuda"),
        )
        map_targets.append(curr_target)
    
    return map_targets