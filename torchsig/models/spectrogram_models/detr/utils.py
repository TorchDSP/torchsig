import torch
import numpy as np
from torch import nn
import torch.distributed as dist
from typing import List, Optional
from torchvision.ops.boxes import box_area


def drop_classifier(parent):
    return torch.nn.Sequential(*list(parent.children())[:-2])


def find_output_features(parent, num_features=0):
    for n, m in parent.named_children():
        if type(m) is torch.nn.Conv2d:
            num_features = m.out_channels
        else:
            num_features = find_output_features(m, num_features)
    return num_features

    
def xcit_name_to_timm_name(input_name: str) -> str:
    if 'nano' in input_name:
        model_name = 'xcit_nano_12_p16_224'
    elif 'tiny' in input_name:
        if '24' in input_name:
            model_name = 'xcit_tiny_24_p16_224'
        else:
            model_name = 'xcit_tiny_12_p16_224'
    elif 'small' in input_name:
        model_name = 'xcit_small_24_p8_224'
    elif 'medium' in input_name:
        model_name = 'xcit_medium_24_p8_224'
    elif 'large' in input_name:
        model_name = 'xcit_large_24_p8_224'
    else:
        raise NotImplemented('Input transformer not supported.')
    
    return model_name


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


def format_preds(preds):
    map_preds = []
    for (i, (det_logits, det_boxes)) in enumerate(zip(preds['pred_logits'], preds['pred_boxes'])):
        boxes = []
        scores = []
        labels = []

        # Convert DETR output format to expected bboxes
        num_objs = 0
        pred = {}
        pred['pred_logits'] = det_logits
        pred['pred_boxes'] = det_boxes

        det_list = []
        for obj_idx in range(pred['pred_logits'].shape[0]):
            probs = pred['pred_logits'][obj_idx].softmax(-1)
            max_prob = probs.max().cpu().detach().numpy()
            max_class = probs.argmax().cpu().detach().numpy()
            if max_class != (pred['pred_logits'].shape[1] - 1) and max_prob >= 0.5:
                center_time = pred['pred_boxes'][obj_idx][0]
                center_freq = pred['pred_boxes'][obj_idx][1]
                duration = pred['pred_boxes'][obj_idx][2]
                bandwidth = pred['pred_boxes'][obj_idx][3]

                # Save to box, score, label lists
                x1 = max(0,(center_time - duration / 2) * 512)
                y1 = max(0,(center_freq - bandwidth / 2) * 512)
                x2 = min(512,(center_time + duration / 2) * 512)
                y2 = min(512,(center_freq + bandwidth / 2) * 512)
                
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
            center_time = label["boxes"][label_obj_idx][0]
            center_freq = label["boxes"][label_obj_idx][1]
            duration = label["boxes"][label_obj_idx][2]
            bandwidth = label["boxes"][label_obj_idx][3]
            class_idx = label["labels"][label_obj_idx]
            
            x1 = (center_time - duration / 2) * 512
            y1 = (center_freq - bandwidth / 2) * 512
            x2 = (center_time + duration / 2) * 512
            y2 = (center_freq + bandwidth / 2) * 512
            
            boxes.append([x1, y1, x2, y2])
            labels.extend([int(class_idx)])
            
        curr_target = dict(
            boxes=torch.tensor(boxes).to("cuda"),
            labels=torch.IntTensor(labels).to("cuda"),
        )
        map_targets.append(curr_target)
    
    return map_targets
