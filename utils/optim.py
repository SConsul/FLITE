# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn.functional as F
optimizers = {
        'adam': torch.optim.Adam,
        'sgd' : torch.optim.SGD
        }

def cross_entropy(test_logits, test_labels, reduction='mean'):
    return F.cross_entropy(test_logits, test_labels, reduction=reduction)

def bbox_loss(bbox_pred, bbox_true, eps=1e-9):
    bbox_pred = bbox_pred.T
    bbox_true = bbox_true.T

    b1_x1, b1_x2 = bbox_pred[0] - bbox_pred[2] / 2, bbox_pred[0] + bbox_pred[2] / 2
    b1_y1, b1_y2 = bbox_pred[1] - bbox_pred[3] / 2, bbox_pred[1] + bbox_pred[3] / 2
    b2_x1, b2_x2 = bbox_true[0] - bbox_true[2] / 2, bbox_true[0] + bbox_true[2] / 2
    b2_y1, b2_y2 = bbox_true[1] - bbox_true[3] / 2, bbox_true[1] + bbox_true[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    # Calculate DIoU
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
            (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
    diou = iou - rho2 / c2  # DIoU
    
    return -1.0 * torch.mean(diou)

def init_optimizer(model, lr, optimizer_type='adam', extractor_scale_factor=1.0,bbox_scale_factor=1.0, with_bbox=False):
    feature_extractor_params = list(map(id, model.feature_extractor.parameters()))
    bbox_params = None
    base_params = None
    optimizer_fn = optimizers[optimizer_type]
    if with_bbox and model.classifier.bbox_head is not None:
        bbox_params = list(map(id, model.classifier.bbox_head.parameters()))
        base_params = filter(lambda p: id(p) not in feature_extractor_params and id(p) not in bbox_params, model.parameters())
        optimizer = optimizer_fn([
                        {'params': base_params },
                        {'params': model.feature_extractor.parameters(), 'lr': lr*extractor_scale_factor},
                        {'params': model.classifier.bbox_head.parameters(), 'lr': lr*bbox_scale_factor}
                        ], lr=lr)
    else:
        base_params = filter(lambda p: id(p) not in feature_extractor_params, model.parameters())
    
        optimizer = optimizer_fn([
                            {'params': base_params },
                            {'params': model.feature_extractor.parameters(), 'lr': lr*extractor_scale_factor}
                            ], lr=lr)
    optimizer.zero_grad()
    return optimizer
