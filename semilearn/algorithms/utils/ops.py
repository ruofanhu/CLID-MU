# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np 
from .loss import ce_loss
import torch.nn.functional as F



    
def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def mixup_one_target(x, y, alpha=1.0, is_bias=False):
    """Returns mixed inputs, mixed targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    if is_bias:
        lam = max(lam, 1 - lam)

    index = torch.randperm(x.size(0)).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y, lam

# @torch.no_grad()
# def get_estimated_y_weights(loader_dict,wnet,model,etargets,weights):
    
#     train_loader = loader_dict['train_lb']
#     t_beta = 0.9


#     for idx_lb,x_lb, y_lb,y_true in train_loader:
#         inputs, targets = x_lb.cuda(), y_lb.cuda() 
#         outputs = model(inputs) 
#         logits_x_lb = outputs['logits'] 
#         y_pred = F.softmax(logits_x_lb,dim=1)
#         y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
#         p = y_pred.data.detach()
#         etargets[idx_lb] = t_beta*etargets[idx_lb] + (1-t_beta)*p

#         # all_etargets[idx_lb] = beta * all_etargets[idx_lb] + (1-beta) * ((y_pred_)/(y_pred_).sum(dim=1,keepdim=True))        
#         loss = ce_loss(logits_x_lb, targets,reduction='none')
#         cost_w = torch.reshape(loss, (len(loss), 1))
#         w = wnet(cost_w.data)
#         weights[idx_lb] = w.data
#     # normalize the weights
#     weights = (weights-weights.min())/(weights.max()-weights.min())
    
def get_estimate_targets(idx_lb,logits_x_lb,etargets,t_beta=0.9):
    y_pred = F.softmax(logits_x_lb,dim=1)
    y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
    p = y_pred.data.detach()
    e_y = t_beta*etargets[idx_lb] + (1-t_beta)*p
    etargets[idx_lb] = e_y
    return e_y
    
        
