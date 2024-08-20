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





def TwoNN(X):
    """input: X, features (n*p) p is the #dimnesions"""
    
    N = X.shape[0]
    distance = torch.cdist(X,X, p=2)
    distance_sorted,indices = torch.sort(distance)
    # if torch.cuda.is_available():
    #     mu = torch.zeros(N).cuda()
    # else:
    #     mu = torch.zeros(N)

    selected = distance_sorted[:, 1:3]
    mu = selected[:,1]/selected[:,0]
    
    sorted_mu,idx_sortmu = torch.sort(mu)
    Femp = torch.arange(N).float() / N
    if torch.cuda.is_available():
        Femp =Femp.cuda()
    
    # Y = X.T.dot(w) the w here is the lid 
    x = torch.log(sorted_mu).view(-1, 1)
    y = -torch.log(1 - Femp).view(-1, 1)

    lid = torch.inverse(x.T @ x) @ x.T @ y #slope
    
    return lid

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

def manifold_loss_soft_exp_no1_(feat,logits,contrast_th):

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

def manifold_loss_soft_exp_no1(feat,logits,contrast_th):

    temperature = contrast_th # temperature could affect
    temperature =0.1
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
    
def hard_balance(probs_x_ulb):
    eps= 1e-7
    probs_x_ulb =probs_x_ulb.detach()

    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    n = probs_x_ulb.shape[1]
    y_one=F.one_hot(y, num_classes=n)
    y_one_= y_one+eps
    w_mat=y_one_/y_one_.sum(dim=0)
    w_mat_new = w_mat/w_mat.sum() 

    w = w_mat_new[y_one==1]
    return w
    


def soft_balance(probs_x_ulb):
    eps= 1e-7

    probs_x_ulb = probs_x_ulb.detach()
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    n = probs_x_ulb.shape[1]
    y_one = F.one_hot(y, num_classes=n)
    y_soft = probs_x_ulb*y_one
    y_soft_= y_soft+eps
    
    w_mat = y_soft_/y_soft_.sum(dim=0)
    w_mat_new = w_mat/w_mat.sum() 
    w = w_mat_new[y_one==1]
    return w
    
def manifold_loss_soft_exp_no1_hard(feat,logits,contrast_th):

    temperature=0.1 # temperature could affect
    probs_x_ulb = torch.softmax(logits, dim=-1)
    # max_probs, y = torch.max(probs_x_ulb, dim=-1)
    feat = F.normalize(feat)
    normalized_probs = F.normalize(probs_x_ulb) #l2norm??
    
    m_feat = feat.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat.shape[0] # n_feat is the # of instances 
    

    sim_n = torch.mm(feat, feat.t())/temperature 
    # sim.fill_diagonal_(1)

    pos_mask = (sim_n>=contrast_th).float()
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
    w=hard_balance(probs_x_ulb)
    out= torch.sum(w*entr)
    return out

def manifold_loss_soft_exp_no1_soft(feat,logits,contrast_th):
   
# kernel: apply kernel trick or not
    temperature=0.1 # temperature could affect
    probs_x_ulb = torch.softmax(logits, dim=-1)
    # max_probs, y = torch.max(probs_x_ulb, dim=-1)
    feat = F.normalize(feat)
    normalized_probs = F.normalize(probs_x_ulb) #l2norm??
    
    m_feat = feat.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat.shape[0] # n_feat is the # of instances 
    

    sim_n = torch.mm(feat, feat.t())/temperature 
    # sim.fill_diagonal_(1)

    pos_mask = (sim_n>=contrast_th).float()
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
    w = soft_balance(probs_x_ulb)
    out= torch.sum(w*entr)
    return out


def entropy_intra(x, logits,th):

    probs_x_ulb = torch.softmax(logits.detach(), dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    mask = max_probs.ge(th).to(max_probs.dtype)
    entr = torch.zeros(10,requires_grad=True).cuda()
    
    y_ = y[mask==1]
    x_ = x[mask==1]
    
    for idx in torch.arange(10):
        _x = x_[y_==idx]
        
        if len(_x) == 0:
            continue
        mean_x = torch.mean(_x, dim=0).reshape(1, -1)
        inner = torch.mm(mean_x, _x.T)
        inner /= torch.linalg.norm(mean_x, dim=1)*torch.linalg.norm(_x.T, dim=0)
        # entr[idx] = (torch.sum(inner) - torch.sum(torch.diagonal(inner, 0)))/2
        entr[idx] = inner.mean()
    return entr.mean()


def entropy_inter(x, logits,th): # or kernel trick
    
    probs_x_ulb = torch.softmax(logits.detach(), dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    mask = max_probs.ge(th).to(max_probs.dtype)
    
    y_ = y[mask==1]
    x_ = x[mask==1]
    # print(torch.unique(y,return_counts=True))
    entr = torch.zeros(1,requires_grad=True).cuda()
    centers = []
    for idx in torch.arange(10):
        _x = x_[y_==idx]
        if len(_x) == 0:
            continue
        mean_x = torch.mean(_x, dim=0)
        
        centers.append(mean_x.reshape(1, -1))
        
    if centers:    
        centers = torch.cat(centers, dim=0)

        inner = torch.mm(centers, centers.T)
        inner /= torch.linalg.norm(centers, dim=1)*torch.linalg.norm(centers.T, dim=0)
        
        entr = (torch.sum(inner) - torch.sum(torch.diagonal(inner, 0)))/2
    return entr


def entropy(x, y,th):
    return entropy_inter(x, y,th), entropy_intra(x, y,th)




def entropy_intra_p(logits,th):
    
    probs_x_ulb = torch.softmax(logits, dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    mask = max_probs.ge(th).to(max_probs.dtype).detach()
    entr = torch.zeros(10,requires_grad=True).cuda()
    
    y_ = y[mask==1].detach()
    # m = torch.nn.BatchNorm1d(10).cuda()
    # logits = m(logits)
    logits = F.normalize(logits,p=2,dim=1)
    
    x_ = logits[mask==1]
    
    for idx in torch.arange(10):
        _x = x_[y_==idx]
        
        if len(_x) == 0:
            continue
        mean_x = torch.mean(_x, dim=0).reshape(1, -1)
        dist = _x - mean_x
        inner = torch.linalg.norm(dist, dim=1)
        # entr[idx] = (torch.sum(inner) - torch.sum(torch.diagonal(inner, 0)))/2
        entr[idx] = inner.mean()
    return entr.mean()


def entropy_inter_p(logits,th): # or kernel trick
    
    probs_x_ulb = torch.softmax(logits, dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    mask = max_probs.ge(th).to(max_probs.dtype)
    
    y_ = y[mask==1]
    x_ = probs_x_ulb[mask==1]
    # print(torch.unique(y,return_counts=True))
    entr = torch.zeros(1,requires_grad=True).cuda()
    centers = []
    for idx in torch.arange(10):
        _x = x_[y_==idx]
        if len(_x) == 0:
            continue
        mean_x = torch.mean(_x, dim=0)
        
        centers.append(mean_x.reshape(1, -1))
        
    if centers:    
        centers = torch.cat(centers, dim=0)
        s1 = centers.clone().reshape(1, centers.shape[0], centers.shape[1])
        s2 = centers.clone().reshape(centers.shape[0], 1, centers.shape[1])        
        entr = torch.linalg.norm(s1 - s2,dim=2)
        
    return entr.mean()/20


def entropy_p(x,th):
    
    return entropy_intra_p(x,th), entropy_inter_p(x,th) 


def entropy_p(feat,logits,th):
    
    with torch.no_grad():
        probs_x_ulb = torch.softmax(logits, dim=-1)
        max_probs, y = torch.max(probs_x_ulb, dim=-1)
        mask = max_probs.ge(th).to(max_probs.dtype)
        y_ = y[mask==1]
        # m = torch.nn.BatchNorm1d(10).cuda()
        # logits = m(logits)
        feat = F.normalize(feat,p=2,dim=1)
        x_ = feat[mask==1]
    
    
    intra_entr = torch.zeros(10,requires_grad=True).cuda()
    # inter_entr = torch.zeros(1,requires_grad=True).cuda()
    centers = []
    

    
    for idx in torch.arange(10):
        _x = x_[y_==idx]
        
        if len(_x) == 0:
            continue
        mean_x = torch.mean(_x, dim=0).reshape(1, -1)
        dist = _x - mean_x
        inner = torch.linalg.norm(dist, dim=1)
        # entr[idx] = (torch.sum(inner) - torch.sum(torch.diagonal(inner, 0)))/2
        intra_entr[idx] = inner.mean()
        
        centers.append(mean_x.reshape(1, -1))
    if centers:    
        centers = torch.cat(centers, dim=0)
        s1 = centers.clone().reshape(1, centers.shape[0], centers.shape[1])
        s2 = centers.clone().reshape(centers.shape[0], 1, centers.shape[1])        
        inter_entr = torch.linalg.norm(s1 - s2,dim=2)
    
    return intra_entr.mean(), inter_entr.sum()/44
    
    
def manifold_loss_hard(feat,logits,th=0.7,kernel=False):

    sigma = 1.1
    probs_x_ulb = torch.softmax(logits, dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    mask = max_probs.ge(th).to(max_probs.dtype)
    
    y_ = y[mask==1].detach()
    logits_ = logits[mask==1]
    feat_ = feat[mask==1]
    feat_ = F.normalize(feat_,p=2,dim=1)

    m_feat = feat_.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat_.shape[0] # n_feat is the # of instances 

    s1 = feat_.clone().reshape(1, n_feat, m_feat)
    s2 = feat_.clone().reshape(n_feat, 1, m_feat)
    # ind = torch.where((torch.max(s1, dim = 2).indices - torch.max(s2, dim = 2).indices) == 0)
    ind = torch.where(y_.reshape(1,-1)==y_.reshape(-1,1))
    
    s_ij =  -torch.ones(n_feat, n_feat).cuda()
    s_ij[ind] = 1
    
    if kernel:
        
        s_ij1 = torch.exp((-torch.sum((s1 - s2) ** 2, dim = 2)) / (2 * (sigma ** 2)))
    else:
    #raw:
        s_ij1 = torch.linalg.norm(s1 - s2,dim=2)
    # hop-k1 within mainfold, hop-k2 of different manifolds
    assert s_ij.shape==(n_feat,n_feat)
    
    entr = s_ij.detach() * s_ij1
    
    return entr.mean()






def manifold_loss_hard_exp(feat,logits,th,kernel=False):


    sigma = 1.1
    temperature=0.1
    probs_x_ulb = torch.softmax(logits, dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    mask = max_probs.ge(th).to(max_probs.dtype)
    normalized_probs = F.normalize(probs_x_ulb)
    # mask = mask.detach()

    y_ = y[mask==1].detach()
    logits_ = logits[mask==1]
    feat_ = feat[mask==1]
    feat_ = F.normalize(feat_,p=2,dim=1)
    normalized_probs_ = normalized_probs[mask==1]
    
    m_feat = feat_.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat_.shape[0] # n_feat is the # of instances 
    
    sim = torch.mm(feat_, feat_.t())/temperature 
    s_ij1 = F.softmax(sim,dim=-1)
    
    eps= 1e-7
    # weight_graph = torch.mm(max_probs.clone().reshape(-1,1), max_probs.clone().reshape(1,-1))
    weight_graph = torch.mm(normalized_probs_,normalized_probs_.t())
    
    weight_graph_ = weight_graph/weight_graph.sum(1,keepdim=True)
    # hop-k1 within mainfold, hop-k2 of different manifolds
    # assert s_ij.shape==(n_feat,n_feat)
    
    entr = torch.sum(-torch.log(s_ij1+eps)*weight_graph_,dim=1)
    
    return entr.mean()


def manifold_loss_soft_exp_two(feat,logits,kernel=False):

    sigma = 1.1  # tunable sigma
    probs_x_ulb = torch.softmax(logits, dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    feat = F.normalize(feat)
    normalized_probs = F.normalize(probs_x_ulb)
    
    m_feat = feat.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat.shape[0] # n_feat is the # of instances 
    

    sim = torch.mm(feat, feat.t())/temperature 
    s_ij1 = F.softmax(sim,dim=-1)
    
    ####
    weight_graph = torch.mm(normalized_probs,normalized_probs.t())
    
    weight_graph_ = weight_graph/weight_graph.sum(1,keepdim=True)
    # weight_graph_ = F.softmax(weight_graph,dim=-1)

    eps= 1e-7
    # assert s_ij.shape==(n_feat,n_feat)
    assert weight_graph.shape==(n_feat,n_feat)
    
    # entr_1 = torch.sum(-torch.log(weight_graph_+eps)*s_ij1,dim=1)
    # entr_2 = torch.sum(-torch.log(1-weight_graph_+eps)*(1-s_ij1),dim=1)
    
    entr_1 = torch.sum(-torch.log(s_ij1+eps)*weight_graph_,dim=1)
    entr_2 = torch.sum(-torch.log(1-s_ij1+eps)*(1-weight_graph_),dim=1)
    
    entr = entr_1+entr_2
    return entr.mean()


def kl_loss_s(feat,logits):
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

def JSD(feat,logits):
    temperature=0.1
    sigma = 1.1  # tunable sigma    
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
    
    # weight_graph_ = weight_graph/weight_graph.sum(1,keepdim=True)
    weight_graph_ = F.softmax(weight_graph,dim=-1)
    
    assert weight_graph.shape==(n_feat,n_feat)
    
    p_mixture = torch.clamp(( weight_graph_ +s_ij1) / 2., 1e-7, 1).log()
    
    entr = F.kl_div(p_mixture,weight_graph_,reduction='batchmean') + F.kl_div( p_mixture,s_ij1,reduction='batchmean')
    
    
    
    return entr/2



def manifold_raw(logits,th,kernel=False):
    
    sigma = 1.1
    probs = torch.softmax(logits, dim=-1)
    max_probs, y = torch.max(probs, dim=-1)
    mask = max_probs.ge(th).to(max_probs.dtype)
    # mask = mask.detach()

    # logits = F.normalize(logits,p=2,dim=1)
    # y_ = y[mask==1].detach()
    logits_ = logits[mask==1]
    probs_ = logits[mask==1]
    
    s1 = probs_.clone().reshape(1, probs_.shape[0], probs_.shape[1])
    s2 = probs_.clone().reshape(probs_.shape[0], 1, probs_.shape[1])
    ind = torch.where((torch.max(s1, dim = 2).indices - torch.max(s2, dim = 2).indices) == 0)
    s_ij =  -torch.ones(probs_.shape[0], probs_.shape[0]).cuda()
    #s_ij = torch.zeros(clean.shape[0], clean.shape[0]).to(device)
    s_ij[ind] = 1
    if kernel:
        s_ij1 = torch.exp((-torch.sum((s1 - s2) ** 2, dim = 2)) / (2 * (args.sigma ** 2)))
    else:
        s_ij1 = torch.linalg.norm(s1 - s2,dim=2)
    
    entr = torch.mean(s_ij.detach() * s_ij1)
    
    return entr




def manifold_hop(feat,logits,th,kernel=False):
    
    sigma = 1.1
    probs_x_ulb = torch.softmax(logits, dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    mask = max_probs.ge(th).to(max_probs.dtype)
    
    y_ = y[mask==1]
    logits_ = logits[mask==1]
    feat_ = feat[mask==1]
    feat_ = F.normalize(feat_,p=2,dim=1)

    m_feat = feat_.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat_.shape[0] # n_feat is the # of instances 

    s1 = feat_.clone().reshape(1, n_feat, m_feat)
    s2 = feat_.clone().reshape(n_feat, 1, m_feat)
    # ind = torch.where((torch.max(s1, dim = 2).indices - torch.max(s2, dim = 2).indices) == 0)
    ind = torch.where(y_.reshape(1,-1)==y_.reshape(-1,1))
    
    s_ij =  -torch.ones(n_feat, n_feat).cuda()
    s_ij[ind] = 1
    
    if kernel:
        
        s_ij1 = torch.exp((-torch.sum((s1 - s2) ** 2, dim = 2)) / (2 * (sigma ** 2)))
    else:
    #raw:
        s_ij1 = torch.linalg.norm(s1 - s2,dim=2)
    # hop-k1 within mainfold, hop-k2 of different manifolds
    assert s_ij.shape==(n_feat,n_feat)
    
    entr = s_ij.detach() * s_ij1
    
    return entr




def pc_loss(feat, logits,class_prototype,temperature):
    
    _, labels =  torch.max(logits, dim=-1).detach()
    probs = torch.softmax(logits, dim=-1).detach()
    
    logits_proto = torch.mm(feat, class_prototype.t())/temperature
    if hard:
        loss_proto = -torch.mean(torch.sum(F.log_softmax(logits_proto, dim=1) * F.one_hot(labels), dim=1))
    else:
        loss_proto = -torch.mean(torch.sum(F.log_softmax(logits_proto, dim=1) * probs, dim=1))
        
    return loss_proto


def gradient_match(loss1,loss2,net):

    grad_sim1=[]
    for n, p in net.named_parameters():
        # if len(p.shape) == 1: continue

        grad1 = grad([loss1],
                     [p],
                     create_graph=True,
                     only_inputs=True,
                     allow_unused=False)[0]
        grad2 = grad([loss2],
                     [p],
                     create_graph=True,
                     only_inputs=True,
                     allow_unused=False)[0]
        _cossim = torch.mm(grad1,grad2.t())
       
        # if len(p.shape) > 1:
        #     _cossim = F.cosine_similarity(grad1, grad2, dim=1).mean()
        # else:
        #     _cossim = F.cosine_similarity(grad1, grad2, dim=0)
        #_mse = F.mse_loss(fake_grad, real_grad)
        grad_cossim11.append(_cossim)
        #grad_mse.append(_mse)

    grad_cossim1 = torch.stack(grad_cossim11)
    gm_loss = grad_cossim1.mean()
    # gm_loss1 = (1.0 - grad_cossim1).mean()
    return gm_loss





def manifold_loss_soft_exp_no1r(feat,logits,contrast_th):

    temperature=0.1
    probs_x_ulb = torch.softmax(logits, dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    feat = F.normalize(feat)
    normalized_probs = F.normalize(probs_x_ulb)
    
    m_feat = feat.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat.shape[0] # n_feat is the # of instances 
    

    sim_n = torch.mm(feat, feat.t())/temperature 
    # sim.fill_diagonal_(1) 
    pos_mask = (sim_n>=contrast_th).float()
    sim_n = sim_n * pos_mask
    s_ij1 = F.softmax(sim_n,dim=-1)
    ####
    eps= 1e-7
    # weight_graph = torch.mm(max_probs.clone().reshape(-1,1), max_probs.clone().reshape(1,-1))
    weight_graph = torch.mm(normalized_probs,normalized_probs.t())
    
    weight_graph_ = weight_graph/weight_graph.sum(1,keepdim=True)

    assert weight_graph.shape==(n_feat,n_feat)
    entr = torch.sum(-torch.log(weight_graph_)*s_ij1.detach(),dim=1)
    return entr.mean()






def manifold_loss_soft_exp_no1r_(feat,logits,contrast_th):

    temperature=0.1
    probs_x_ulb = torch.softmax(logits, dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    feat = F.normalize(feat)
    normalized_probs = F.normalize(probs_x_ulb)
    
    m_feat = feat.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat.shape[0] # n_feat is the # of instances 
    

    sim_n = torch.mm(feat, feat.t())/temperature 
    # sim.fill_diagonal_(1) 
    pos_mask = (sim_n>=contrast_th).float()
    sim_n_ = sim_n * pos_mask
    s_ij1 = F.softmax(sim_n_,dim=-1)
    
    ####
    eps= 1e-7
    # weight_graph = torch.mm(max_probs.clone().reshape(-1,1), max_probs.clone().reshape(1,-1))
    weight_graph = torch.mm(normalized_probs,normalized_probs.t())
    
    weight_graph_ = weight_graph/weight_graph.sum(1,keepdim=True).detach()

    assert weight_graph.shape==(n_feat,n_feat)

    
    # entr = torch.sum(-torch.log(s_ij1.detach()+eps)*weight_graph_,dim=1)
    entr = torch.sum(-torch.log(weight_graph_)*s_ij1,dim=1)

    
    return entr.mean()


def manifold_loss_soft_exp(feat,logits,contrast_th):

    temperature=0.1
    probs_x_ulb = torch.softmax(logits, dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    feat = F.normalize(feat)
    normalized_probs = F.normalize(probs_x_ulb)
    
    m_feat = feat.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat.shape[0] # n_feat is the # of instances 
    

    sim_n = torch.mm(feat, feat.t())/temperature 
    sim_n.fill_diagonal_(1) 
    pos_mask = (sim_n>=contrast_th).float()
    sim_n = sim_n * pos_mask
    s_ij1 = F.softmax(sim_n,dim=-1)
    
    ####
    eps= 1e-7
    # weight_graph = torch.mm(max_probs.clone().reshape(-1,1), max_probs.clone().reshape(1,-1))
    weight_graph = torch.mm(normalized_probs,normalized_probs.t())
    
    weight_graph_ = weight_graph/weight_graph.sum(1,keepdim=True)

    assert weight_graph.shape==(n_feat,n_feat)

    
    entr = torch.sum(-torch.log(s_ij1.detach()+eps)*weight_graph_,dim=1)

    
    return entr.mean()


def manifold_loss_soft_expr(feat,logits,contrast_th):

    temperature=0.1
    probs_x_ulb = torch.softmax(logits, dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    feat = F.normalize(feat)
    normalized_probs = F.normalize(probs_x_ulb)
    
    m_feat = feat.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat.shape[0] # n_feat is the # of instances 
    

    sim_n = torch.mm(feat, feat.t())/temperature 
    sim_n.fill_diagonal_(1) 
    pos_mask = (sim_n>=contrast_th).float()
    sim_n = sim_n * pos_mask
    s_ij1 = F.softmax(sim_n,dim=-1)
    
    ####
    eps= 1e-7
    # weight_graph = torch.mm(max_probs.clone().reshape(-1,1), max_probs.clone().reshape(1,-1))
    weight_graph = torch.mm(normalized_probs,normalized_probs.t())
    
    weight_graph_ = weight_graph/weight_graph.sum(1,keepdim=True)

    assert weight_graph.shape==(n_feat,n_feat)

    
    entr = torch.sum(-torch.log(weight_graph_)*s_ij1.detach(),dim=1)

    
    return entr.mean()


def manifold_loss_soft_exp_(feat,logits,contrast_th):

    temperature=0.1
    
    probs_x_ulb = torch.softmax(logits, dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    feat = F.normalize(feat)
    normalized_probs = F.normalize(probs_x_ulb)
    
    m_feat = feat.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat.shape[0] # n_feat is the # of instances 
    
    sim_n = torch.mm(feat, feat.t())/temperature 
    sim_n.fill_diagonal_(1) 
    pos_mask = (sim_n>=contrast_th).float()
    sim_n = sim_n * pos_mask
    s_ij1 = F.softmax(sim_n,dim=-1)
    ####
    eps= 1e-7
    # weight_graph = torch.mm(max_probs.clone().reshape(-1,1), max_probs.clone().reshape(1,-1))
    weight_graph = torch.mm(normalized_probs,normalized_probs.t())
    weight_graph_ = weight_graph/weight_graph.sum(1,keepdim=True).detach()

    assert weight_graph_.shape==(n_feat,n_feat)

    entr = torch.sum(-torch.log(s_ij1.detach()+eps)*weight_graph_,dim=1)
    # entr = torch.sum(-s_ij1*torch.log(weight_graph_+eps),dim=1)

    
    return entr.mean()





def manifold_loss_soft_expr_(feat,logits,contrast_th):

    temperature=0.1
    sigma = 1.1  # tunable sigma
    probs_x_ulb = torch.softmax(logits, dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    feat = F.normalize(feat)
    normalized_probs = F.normalize(probs_x_ulb)
    
    m_feat = feat.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat.shape[0] # n_feat is the # of instances 
    
    sim_n = torch.mm(feat, feat.t())/temperature 
    sim_n.fill_diagonal_(1) 
    pos_mask = (sim_n>=contrast_th).float()
    sim_n = sim_n * pos_mask
    s_ij1 = F.softmax(sim_n,dim=-1)
    # s_ij1 = F.softmax(sim,dim=-1)
    
    ####
    eps= 1e-7
    # weight_graph = torch.mm(max_probs.clone().reshape(-1,1), max_probs.clone().reshape(1,-1))
    weight_graph = torch.mm(normalized_probs,normalized_probs.t())
    
    weight_graph_ = weight_graph/weight_graph.sum(1,keepdim=True).detach()
    # weight_graph_ = F.softmax(weight_graph,dim=-1)

    # assert s_ij.shape==(n_feat,n_feat)
    assert weight_graph.shape==(n_feat,n_feat)

    
    entr = torch.sum(-torch.log(weight_graph_)*s_ij1.detach(),dim=1)
    # entr = torch.sum(-s_ij1*torch.log(weight_graph_+eps),dim=1)

    
    return entr.mean()





def graph_sim_dct_ce(feat,logits,kernel=False):

    temperature=0.2
    contrast_th = 0.8
    probs_x_ulb = torch.softmax(logits, dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    # feat = F.normalize(feat)
    # normalized_probs = F.normalize(probs_x_ulb)
    
    m_feat = feat.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat.shape[0] # n_feat is the # of instances 
    
    sim = torch.mm(feat, feat.t())/temperature 
    sim.fill_diagonal_(1) 
    pos_mask = (sim>=contrast_th).float()
    sim= sim*pos_mask
    s_ij1 = F.softmax(sim,dim=-1)
    ####
    eps= 1e-7
    # weight_graph = torch.mm(max_probs.clone().reshape(-1,1), max_probs.clone().reshape(1,-1))
    weight_graph = torch.mm(probs_x_ulb,probs_x_ulb.t())
    weight_graph.fill_diagonal_(1)   
    
    # weight_graph = weight_graph * pos_mask
    
    weight_graph_ = weight_graph/weight_graph.sum(1,keepdim=True).detach()

    assert weight_graph_.shape==(n_feat,n_feat)

    entr = torch.sum(-torch.log(s_ij1.detach()+eps)*weight_graph_,dim=1)
    # entr = torch.sum(-s_ij1*torch.log(weight_graph_+eps),dim=1)

    
    return entr.mean()

def graph_sim_dct_ce_n(feat,logits,contrast_th=0.8):

    temperature=1
    # contrast_th = 0.8
    probs_x_ulb = torch.softmax(logits, dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    feat_n = F.normalize(feat)
    # probs_x_ulb = F.normalize(probs_x_ulb)
    
    m_feat = feat.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat.shape[0] # n_feat is the # of instances 
    
    # sim = torch.mm(feat, feat.t())/temperature 
    sim_n = torch.mm(feat_n, feat_n.t())/temperature 
    # sim.fill_diagonal_(1)   
    pos_mask = (sim_n>=contrast_th).float()
    sim_n = sim_n * pos_mask
    s_ij1 = F.softmax(sim_n,dim=-1)
    ####
    eps= 1e-7
    # weight_graph = torch.mm(max_probs.clone().reshape(-1,1), max_probs.clone().reshape(1,-1))
    weight_graph = torch.mm(probs_x_ulb,probs_x_ulb.t()) # entries rangre from [0,1]

    
    weight_graph_ = weight_graph/weight_graph.sum(1,keepdim=True).detach()

    assert weight_graph_.shape==(n_feat,n_feat)

    entr = torch.sum(-torch.log(s_ij1.detach()+eps)*weight_graph_,dim=1)
    # entr = torch.sum(-s_ij1.detach()*torch.log(weight_graph_+eps),dim=1)
    
    return entr.mean()

def graph_sim_dct_ce_nr(feat,logits,contrast_th=0.8):

    temperature=1
    # contrast_th = 0.8
    probs_x_ulb = torch.softmax(logits, dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    feat_n = F.normalize(feat)
    # probs_x_ulb = F.normalize(probs_x_ulb)
    
    m_feat = feat.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat.shape[0] # n_feat is the # of instances 
    
    # sim = torch.mm(feat, feat.t())/temperature 
    sim_n = torch.mm(feat_n, feat_n.t())/temperature 
    # sim.fill_diagonal_(1)   
    pos_mask = (sim_n>=contrast_th).float()
    sim_n = sim_n * pos_mask
    s_ij1 = F.softmax(sim_n,dim=-1)
    ####
    eps= 1e-7
    # weight_graph = torch.mm(max_probs.clone().reshape(-1,1), max_probs.clone().reshape(1,-1))
    weight_graph = torch.mm(probs_x_ulb,probs_x_ulb.t()) # entries rangre from [0,1]

    
    weight_graph_ = weight_graph/weight_graph.sum(1,keepdim=True).detach()

    assert weight_graph_.shape==(n_feat,n_feat)

    # entr = torch.sum(-torch.log(s_ij1.detach()+eps)*weight_graph_,dim=1)
    entr = torch.sum(-s_ij1.detach()*torch.log(weight_graph_+eps),dim=1)
    
    return entr.mean()


def graph_sim_dct(feat,logits,kernel=False):

    temperature=0.2
    contrast_th = 0.8
    
    probs_x_ulb = torch.softmax(logits, dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    # feat = F.normalize(feat)
    # normalized_probs = F.normalize(probs_x_ulb)
    
    m_feat = feat.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat.shape[0] # n_feat is the # of instances 
    
    sim = torch.mm(feat, feat.t())/temperature  # considering sim.fill_diagonal_(0) 
    # sim.fill_diagonal_(0) 
    s_ij1 = F.softmax(sim,dim=-1)
    ####
    eps= 1e-7
    # weight_graph = torch.mm(max_probs.clone().reshape(-1,1), max_probs.clone().reshape(1,-1))
    weight_graph = torch.mm(probs_x_ulb,probs_x_ulb.t())
    weight_graph.fill_diagonal_(1)   
    pos_mask = (weight_graph>=contrast_th).float()
    weight_graph = weight_graph * pos_mask
    weight_graph = weight_graph/weight_graph.sum(1,keepdim=True).detach()

    assert weight_graph.shape==(n_feat,n_feat)


    
    entr_sim = torch.sum(s_ij1.detach()*weight_graph,dim=1)
    entr = torch.ones_like(entr_sim)-entr_sim
    # entr = torch.sum(-s_ij1*torch.log(weight_graph_+eps),dim=1)
    # probs_x_ulb_c = probs_x_ulb*pos_mask
    # approx_y =  torch.mm(s_ij1, probs_x_ulb_c)
    
    return entr.mean(), [probs_x_ulb_c.cpu().detach(), s_ij1.cpu().detach()]


def graph_sim_dct_n(feat,logits,kernel=False):

    temperature=0.2
    contrast_th = 0.8
    
    probs_x_ulb = torch.softmax(logits, dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    # feat = F.normalize(feat)
    # normalized_probs = F.normalize(probs_x_ulb)
    
    m_feat = feat.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat.shape[0] # n_feat is the # of instances 
    
    sim = torch.mm(feat, feat.t())/temperature  # considering sim.fill_diagonal_(0) 
    # sim.fill_diagonal_(0) 
    s_ij1 = F.softmax(sim,dim=-1)
    ####
    eps= 1e-7
    # weight_graph = torch.mm(max_probs.clone().reshape(-1,1), max_probs.clone().reshape(1,-1))
    weight_graph = torch.mm(probs_x_ulb,probs_x_ulb.t())
    weight_graph.fill_diagonal_(1)   
    pos_mask = (weight_graph>=contrast_th).float()
    weight_graph = weight_graph * pos_mask
    # print(weight_graph.sum(1,keepdim=True).detach())
    weight_graph = weight_graph/weight_graph.sum(1,keepdim=True).detach()

    assert weight_graph.shape==(n_feat,n_feat)


    
    entr_sim = torch.sum(s_ij1.detach()*weight_graph,dim=1)
    entr = torch.ones_like(entr_sim)-entr_sim
    # entr = torch.sum(-s_ij1*torch.log(weight_graph_+eps),dim=1)
    # approx_y =  torch.mm(s_ij1, probs_x_ulb_c)
    s_masked = s_ij1*pos_mask
    out_= torch.mm(s_masked,probs_x_ulb)
    
    return entr.mean(), out_.cpu().detach()


def manifold_loss_soft(feat,logits,kernel=False):

    contrast_th = 0.8
    sigma = 1.1  # tunable sigma
    probs_x_ulb = torch.softmax(logits, dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    # feat = F.normalize(feat,p=2,dim=1)

    m_feat = feat.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat.shape[0] # n_feat is the # of instances 
    
    s1 = feat.clone().reshape(1, n_feat, m_feat)
    s2 = feat.clone().reshape(n_feat, 1, m_feat)
    # ind = torch.where((torch.max(s1, dim = 2).indices - torch.max(s2, dim = 2).indices) == 0)
    ind = torch.where(y.reshape(1,-1)==y.reshape(-1,1))

    
    s_ij =  -torch.ones(n_feat, n_feat).cuda()
    s_ij[ind] = 1
    
    if kernel:
        
        s_ij1 = torch.exp((-torch.sum((s1 - s2) ** 2, dim = 2)) / (2 * (sigma ** 2)))
    else:
    #raw:
        s_ij1 = torch.linalg.norm(s1 - s2,dim=2)
    # hop-k1 within mainfold, hop-k2 of different manifolds
    
    # weight_graph = torch.mm(max_probs.clone().reshape(-1,1), max_probs.clone().reshape(1,-1))
    weight_graph = torch.mm(probs_x_ulb.clone().reshape(-1,probs_x_ulb.shape[1]), probs_x_ulb.clone().reshape(probs_x_ulb.shape[1],-1))
    weight_graph.fill_diagonal_(1)   
    pos_mask = (weight_graph>=contrast_th).float()
    weight_graph = weight_graph * pos_mask
    weight_graph = weight_graph/weight_graph.sum(1,keepdim=True).detach()
    
    assert s_ij.shape==(n_feat,n_feat)
    assert weight_graph.shape==(n_feat,n_feat)
    
    # entr = weight_graph.detach() * s_ij.detach() * s_ij1  
    entr = weight_graph * s_ij1.detach()  
    # print(weight_graph.detach(), s_ij1)

    return entr.mean()


def manifold_loss_soft_exp_ws_(feat_w,feat_s,logits_w,logits_s,contrast_th):

    temperature=0.1
    probs_x_ulb_w = torch.softmax(logits_w, dim=-1)
    probs_x_ulb_s = torch.softmax(logits_s, dim=-1)
    # max_probs, y = torch.max(probs_x_ulb, dim=-1)
    feat_w = F.normalize(feat_w)
    feat_s = F.normalize(feat_s)
    normalized_probs_w = F.normalize(probs_x_ulb_w)
    normalized_probs_s = F.normalize(probs_x_ulb_s)
    
    m_feat = feat_w.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat_w.shape[0] # n_feat is the # of instances 
    

    sim_n = torch.mm(feat_w, feat_s.t()) 
    sim_n = torch.mm(feat_w, feat_w.t())
    sim_s = torch.mm(feat_w, feat_s.t())
    sim_n[range(len(sim_s)), range(len(sim_s))] = torch.diagonal(sim_s, 0)
    
    sim_n=sim_n/temperature
    pos_mask = (sim_n>=contrast_th).float()
    sim_n = sim_n * pos_mask
    s_ij1 = F.softmax(sim_n,dim=-1)
    
    ####
    eps= 1e-7
    # weight_graph = torch.mm(max_probs.clone().reshape(-1,1), max_probs.clone().reshape(1,-1))
    weight_graph = torch.mm(normalized_probs_w,normalized_probs_s.t())
    weight_graph.fill_diagonal_(1)
    weight_graph_ = weight_graph/weight_graph.sum(1,keepdim=True).detach()

    assert weight_graph.shape==(n_feat,n_feat)

    
    entr = torch.sum(-torch.log(s_ij1.detach()+eps)*weight_graph_,dim=1)

    
    return entr.mean()

def manifold_loss_soft_exp_ws(feat_w,feat_s,logits_w,logits_s,contrast_th):

    temperature=0.1
    probs_x_ulb_w = torch.softmax(logits_w, dim=-1)
    probs_x_ulb_s = torch.softmax(logits_s, dim=-1)
    # max_probs, y = torch.max(probs_x_ulb, dim=-1)
    feat_w = F.normalize(feat_w)
    feat_s = F.normalize(feat_s)
    normalized_probs_w = F.normalize(probs_x_ulb_w)
    normalized_probs_s = F.normalize(probs_x_ulb_s)
    
    m_feat = feat_w.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat_w.shape[0] # n_feat is the # of instances 
    

    sim_n = torch.mm(feat_w, feat_w.t())
    sim_s = torch.mm(feat_w, feat_s.t())
    sim_n[range(len(sim_s)), range(len(sim_s))] = torch.diagonal(sim_s, 0)
    # sim_n.fill_diagonal_(1)
    sim_n=sim_n/temperature
    pos_mask = (sim_n>=contrast_th).float()
    sim_n = sim_n * pos_mask
    s_ij1 = F.softmax(sim_n,dim=-1)
    
    ####
    eps= 1e-7
    # weight_graph = torch.mm(max_probs.clone().reshape(-1,1), max_probs.clone().reshape(1,-1))
    weight_graph = torch.mm(normalized_probs_w,normalized_probs_s.t())
    weight_graph.fill_diagonal_(1)
    weight_graph_ = weight_graph/weight_graph.sum(1,keepdim=True)

    assert weight_graph.shape==(n_feat,n_feat)

    
    entr = torch.sum(-torch.log(s_ij1.detach()+eps)*weight_graph_,dim=1)

    
    return entr.mean()



def dmanifold_loss_soft_exp_no1(feat,logits,contrast_th):

    temperature=0.1
    probs_x_ulb = torch.softmax(logits, dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    feat = F.normalize(feat)
    normalized_probs = F.normalize(probs_x_ulb)
    
    m_feat = feat.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat.shape[0] # n_feat is the # of instances 
    

    sim_n = torch.mm(feat, feat.t())/temperature 
    # sim.fill_diagonal_(1) 
    pos_mask = (sim_n>=contrast_th).float()
    sim_n = sim_n * pos_mask
    s_ij1 = F.softmax(sim_n,dim=-1)
    
    ####
    eps= 1e-7
    # weight_graph = torch.mm(max_probs.clone().reshape(-1,1), max_probs.clone().reshape(1,-1))
    weight_graph = torch.mm(normalized_probs,normalized_probs.t())
    
    weight_graph_ = weight_graph/weight_graph.sum(1,keepdim=True)

    assert weight_graph.shape==(n_feat,n_feat)
    entr = torch.sum(-torch.log(s_ij1+eps)*weight_graph_.detach(),dim=1)

    
    return entr.mean()


def dmanifold_loss_soft_exp_no1_(feat,logits,contrast_th):

    temperature=0.1
    probs_x_ulb = torch.softmax(logits, dim=-1)
    max_probs, y = torch.max(probs_x_ulb, dim=-1)
    feat = F.normalize(feat)
    normalized_probs = F.normalize(probs_x_ulb)
    
    m_feat = feat.shape[1] # m_feat is the # of dimensions of the features
    n_feat = feat.shape[0] # n_feat is the # of instances 
    

    sim_n = torch.mm(feat, feat.t())/temperature 
    # sim.fill_diagonal_(1)

    pos_mask = (sim_n>=contrast_th).float()
    # print(torch.sum((sim_n>=0).float()))
    sim_n = sim_n * pos_mask
    s_ij1 = F.softmax(sim_n,dim=-1)
    
    ####
    eps= 1e-7
    # weight_graph = torch.mm(max_probs.clone().reshape(-1,1), max_probs.clone().reshape(1,-1))
    weight_graph = torch.mm(normalized_probs,normalized_probs.t())
    
    weight_graph_ = weight_graph/weight_graph.sum(1,keepdim=True).detach()

    assert weight_graph.shape==(n_feat,n_feat)

    
    entr = torch.sum(-torch.log(s_ij1+eps)*weight_graph_.detach(),dim=1)

    
    return entr.mean()




