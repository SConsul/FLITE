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
    
def init_optimizer(model, lr, optimizer_type='adam', extractor_scale_factor=1.0,bbox_scale_factor=1.0):
    feature_extractor_params = list(map(id, model.feature_extractor.parameters()))
    bbox_params = None
    base_params = None
    optimizer_fn = optimizers[optimizer_type]
    if model.bbox_head is not None:
        bbox_params = list(map(id, model.bbox_head.parameters()))
        base_params = filter(lambda p: id(p) not in feature_extractor_params and id(p) not in bbox_params, model.parameters())
        optimizer = optimizer_fn([
                        {'params': base_params },
                        {'params': model.feature_extractor.parameters(), 'lr': lr*extractor_scale_factor},
                        {'params': model.bbox_head.parameters(), 'lr': lr*bbox_scale_factor}
                        ], lr=lr)
    else:
        base_params = filter(lambda p: id(p) not in feature_extractor_params, model.parameters())
    
        optimizer = optimizer_fn([
                            {'params': base_params },
                            {'params': model.feature_extractor.parameters(), 'lr': lr*extractor_scale_factor}
                            ], lr=lr)
    optimizer.zero_grad()
    return optimizer
