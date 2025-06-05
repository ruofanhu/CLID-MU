# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch 
import torch.nn as nn 
from torch.nn import functional as F


def metric_smoothing(list_metrics,smoothing_alpha=0.999):
    
    s_list_metrics=[]
    p_l=0
    for i in range(len(list_metrics)):
        p_l = smoothing_alpha *p_l + (1 - smoothing_alpha)* list_metrics[i]
        s_list_metrics.append(p_l/(1 - smoothing_alpha**(i+1)))
    return s_list_metrics

def smooth_targets(logits, targets, smoothing=0.1):
    """
    label smoothing
    """
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(smoothing / (logits.shape[-1] - 1))
        true_dist.scatter_(1, targets.data.unsqueeze(1), (1 - smoothing))
    return true_dist


def ce_loss(logits, targets, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.

    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
#     print(targets,logits)
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
#         print('---------------',logits.shape)
        if logits.shape[1]==1:
#             log_pred = F.sigmoid(logits)
            return F.binary_cross_entropy(torch.sigmoid(logits), targets.reshape([-1,1]).type(torch.float),reduction = reduction)
        else:
            log_pred = F.log_softmax(logits, dim=-1)
            return F.nll_loss(log_pred, targets, reduction=reduction)


def gce_loss(logits, targets, q=0.7,reduction='none'):
    num_classes =  logits.shape[1]
    log_pred = F.log_softmax(logits, dim=-1)
    log_pred = torch.clamp(log_pred, min=1e-7, max=1.0)
    if logits.shape != targets.shape:
        # conver to one-hot target
        targets = torch.nn.functional.one_hot(targets, num_classes).float().to(logits.device)
    
    gce_loss = (1. - torch.pow(torch.sum(targets * log_pred, dim=1), q)) / q
    
    if reduction == 'none':
        return gce_loss
    else:
        return gce_loss.mean()
        
def mae_loss(logits, targets, reduction='none'):
    num_classes =  logits.shape[1]
    pred = F.softmax(logits, dim=-1)
    if logits.shape != targets.shape:
        # conver to one-hot target
        targets = torch.nn.functional.one_hot(targets, num_classes).float().to(logits.device)
    
    mae_loss = 1.0 - torch.sum(targets * pred, dim=1)
    
    if reduction == 'none':
        return mae_loss
    else:
        return mae_loss.mean()



        
def consistency_loss(logits, targets, name='ce', mask=None):
    """
    wrapper for consistency regularization loss in semi-supervised learning.

    Args:
        logits: logit to calculate the loss on and back-propagion, usually being the strong-augmented unlabeled samples
        targets: pseudo-labels (either hard label or soft label)
        name: use cross-entropy ('ce') or mean-squared-error ('mse') to calculate loss
        mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
    """

    assert name in ['ce', 'mse']
    # logits_w = logits_w.detach()
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    else:
        loss = ce_loss(logits, targets, reduction='none')

    if mask is not None:
        # mask must not be boolean type
        loss = loss * mask

    return loss.mean()

def lid(logits,k,distmetric = 'l2'):
    """the input logits could be logits or feature, typically it's logits (howeverbased on the theory it should be feature)"""
    k = min(k, logits.shape[0] - 1)
    if distmetric == 'cos':
        a_norm = logits / logits.norm(p=2,dim=1)[:, None]
        cos_sim = torch.mm(a_norm, a_norm.transpose(0,1))

        cos_distance = torch.ones(cos_sim.size()).cuda() - cos_sim
        distance_sorted,indices = torch.sort(cos_distance)
#         print('cos_dis',cos_sim[cos_sim>1]


    else:
        distance = torch.cdist(logits,logits, p=2)
        distance_sorted,indices = torch.sort(distance)
    selected = distance_sorted[:, 1:k + 1]
    #######
#     selected[selected==0]+=1e-12
    # selected += 1e-12
    # lids = -k/torch.sum(torch.log(selected[:,:-2]/(selected[:,-1].detach() +1e-12).reshape(-1,1)),axis=1)
    lids = -k/torch.sum(torch.log(selected[:,:-2]/(selected[:,-1] +1e-12).reshape(-1,1)),axis=1)
    ######################
#     lids = -k/torch.sum(torch.log(selected/(selected[:,-1]).reshape(-1,1)),axis=1)
#     lids = -k/torch.sum(torch.log(selected/(selected[:,-1]+1e-12).reshape(-1,1)),axis=1)
    return lids.mean()







def variance_loss(x):
    """input: x, features (n*p) p is the #dimnesions"""
    num_features= x.shape[1]

    x = x - x.mean(dim=0)
    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_x))

    cov_x = (x.T @ x) / (x.shape[0] - 1) #cov_x = (x.T @ x) / (self.args.batch_size - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(num_features)

    return std_loss, cov_loss

def exclude_bias_and_norm(p):
    return p.ndim == 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def clid_loss(feat,logits,contrast_th):

    if contrast_th is not None:
        temperature = contrast_th # temperature could affect
    else:
        temperature=0.1
    probs_x_ulb = torch.softmax(logits, dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    feat = F.normalize(feat)
    normalized_probs = F.normalize(probs_x_ulb) #l2norm??
    
    m_feat = feat.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat.shape[0] # n_feat is the # of instances 
    

    sim_n = torch.mm(feat, feat.t())/temperature 
    # sim.fill_diagonal_(1)

    pos_mask = (sim_n>=0).float()
    # print(torch.sum((sim_n>=0).float()))
    sim_n = sim_n * pos_mask
    s_ij1 = F.softmax(sim_n,dim=-1)
    
    ####
    eps= 1e-7
    # weight_graph = torch.mm(max_probs.clone().reshape(-1,1), max_probs.clone().reshape(1,-1))
    weight_graph = torch.mm(normalized_probs,normalized_probs.t())
    
    weight_graph_ = weight_graph/weight_graph.sum(1,keepdim=True).detach()

    assert weight_graph.shape==(n_feat,n_feat)

    entr = torch.sum(-torch.log(s_ij1.detach()+eps)*weight_graph_,dim=1)
    # entr = torch.sum(-torch.log(s_ij1+eps)*weight_graph_,dim=1)

    return entr.mean()




def kl_loss_s(feat,logits):
    temperature=1
    probs_x_ulb = torch.softmax(logits, dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    feat = F.normalize(feat)
    normalized_probs = F.normalize(probs_x_ulb)
    
    m_feat = feat.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat.shape[0] # n_feat is the # of instances 
    

    sim = torch.mm(feat, feat.t())/temperature 
    s_ij1 = F.softmax(sim,dim=-1)
    
    ####
    eps= 1e-7
    weight_graph = torch.mm(normalized_probs,normalized_probs.t())
    
    weight_graph_ = weight_graph/weight_graph.sum(1,keepdim=True)
    # weight_graph_ = F.softmax(weight_graph,dim=-1)

    assert weight_graph.shape==(n_feat,n_feat)


    # torch.nn.functional.kl_div(input, target,log_target=True)
    entr = F.kl_div(weight_graph_,s_ij1,reduction='batchmean') 

    return entr


def kl_loss_w(feat,logits):
    probs_x_ulb = torch.softmax(logits, dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    feat = F.normalize(feat)
    normalized_probs = F.normalize(probs_x_ulb)
    
    m_feat = feat.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat.shape[0] # n_feat is the # of instances 
    

    sim = torch.mm(feat, feat.t())/temperature 
    s_ij1 = F.softmax(sim,dim=-1)
    
    ####
    eps= 1e-7
    weight_graph = torch.mm(normalized_probs,normalized_probs.t())
    
    weight_graph_ = weight_graph/weight_graph.sum(1,keepdim=True)
    # weight_graph_ = F.softmax(weight_graph,dim=-1)

    assert weight_graph.shape==(n_feat,n_feat)


    # torch.nn.functional.kl_div(input, target,log_target=True)
    entr = F.kl_div(s_ij1,weight_graph_,reduction='batchmean') 

    return entr


