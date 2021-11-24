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

def bbox_loss(bbox_pred, bbox_true):
    print("BBOX PRED")
    print(bbox_pred)
    print(bbox_pred.shape)
    print("BBOX TRUE")
    print(bbox_true)
    print(bbox_true.shape)

    loss = torch.mean(
        ((bbox_pred[:, 0] - bbox_true[0])**2) +
        ((bbox_pred[:, 1] - bbox_true[:, 1])**2) +
        ((torch.sqrt(bbox_pred[:, 2]) - torch.sqrt(bbox_true[:, 2]))**2) +
        ((torch.sqrt(bbox_pred[:, 3]) - torch.sqrt(bbox_true[:, 3]))**2)
    )
    print("BBOX LOSS IN LOSS FN")
    print(loss)
    return loss

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
