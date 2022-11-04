import sympy
import numpy as np
import torch
from torch import nn
from scipy import ndimage


def acc(y_hat, y):
    y_hat = y_hat.argmax(1)
    acc = ((y_hat == y)).float().mean()
    return acc


def iou(y_hat, y):
    y_hat = y_hat.argmax(1)
    intersection = ((y_hat == 1) & (y == 1)).sum((1, 2))
    union = ((y_hat == 1) | (y == 1)).sum((1, 2))
    iou = (intersection.float() / union.float()).mean()
    return iou


def class_iou(y_hat, y):
    # print(y_hat.shape) # B, C, H, W
    # print(y.shape) # B, H, W
    y_hat = y_hat.argmax(1)
    # print(y_hat.shape) # B, H, W
    num_classes = 6
    iou = 0
    num_present = 0
    for batch_idx in range(y.shape[0]):
        for class_idx in range(1, num_classes+1):
            if (y == class_idx).float().sum() > 0:
                intersection = ((y_hat == class_idx) & (y == class_idx)).sum((1, 2))
                union = ((y_hat == class_idx) | (y == class_idx)).sum((1, 2))
                class_iou = ((intersection.float() + 1e-6) / (union.float() + 1e-6)).mean()
                iou += class_iou
                num_present += 1
    return iou / num_present


def replace_bn(parent):
    for n, m in parent.named_children():
        if type(m) is nn.BatchNorm2d:
            setattr(
                parent,
                n,
                nn.GroupNorm(
                    min(
                        sympy.divisors(m.num_features),
                        key=lambda x: np.abs(np.sqrt(m.num_features) - x),
                    ),
                    m.num_features,
                ),
            )
        else:
            replace_bn(m)

            
def format_preds(preds, num_classes):
    map_preds = []
    
    # Loop over examples in batch
    for pred in preds:
        boxes = []
        scores = []
        labels = []
        
        # Loop over classes
        for class_idx in range(1,num_classes+1):
            curr_pred = pred.argmax(0)
            curr_indices = (curr_pred == class_idx).cpu().numpy()
            curr_pred = np.zeros((preds.shape[-2], preds.shape[-1]))
            curr_pred[curr_indices] = 1.0
            if curr_pred.sum() == 0:
                continue

            image, num_features = ndimage.label(np.abs(curr_pred))
            objs = ndimage.find_objects(image)

            # # Remove small boxes and append to detected signal object
            # min_dur = 2 # min time duration
            # min_bw = 2 # min bw
            # min_area = 4
            
            for i, ob in enumerate(objs):
                bw = ob[0].stop - ob[0].start
                dur = ob[1].stop - ob[1].start
                # if (dur > min_dur) and (bw > min_bw) and (bw*dur > min_area):
                center_time = (ob[1].stop + ob[1].start) / 2
                center_freq = ob[0].start + bw/2

                boxes.append([ob[1].start, ob[0].start, ob[1].stop, ob[0].stop])
                scores.extend([1.0])
                labels.extend([class_idx-1])
    
        curr_pred = dict(
            boxes=torch.tensor(boxes).to("cuda"),
            scores=torch.tensor(scores).to("cuda"),
            labels=torch.IntTensor(labels).to("cuda"),
        )
        map_preds.append(curr_pred)
    
    return map_preds


def format_targets(targets, num_classes):
    map_targets = []
    
    # Loop over examples in batch
    for target in targets:
        boxes = []
        labels = []
        
        # Loop over classes
        for class_idx in range(1,num_classes+1):
            curr_indices = (target == class_idx).cpu().numpy()
            curr_target = np.zeros((targets.shape[-2], targets.shape[-1]))
            curr_target[curr_indices] = 1.0
            if curr_target.sum() == 0:
                continue

            image, num_features = ndimage.label(np.abs(curr_target))
            objs = ndimage.find_objects(image)

            for i, ob in enumerate(objs):
                bw = ob[0].stop - ob[0].start
                dur = ob[1].stop - ob[1].start
                center_time = (ob[1].stop + ob[1].start) / 2
                center_freq = ob[0].start + bw/2

                boxes.append([ob[1].start, ob[0].start, ob[1].stop, ob[0].stop])
                labels.extend([class_idx-1])
    
        curr_target = dict(
            boxes=torch.tensor(boxes).to("cuda"),
            labels=torch.IntTensor(labels).to("cuda"),
        )
        map_targets.append(curr_target)
    
    return map_targets
