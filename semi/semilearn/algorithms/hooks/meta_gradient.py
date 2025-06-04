# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import copy 
from torch.nn import functional as F

from semilearn.core.hooks import Hook
from semilearn.algorithms.utils.loss import *
from sklearn.metrics import confusion_matrix,roc_auc_score,f1_score,accuracy_score, balanced_accuracy_score
from torcheval.metrics import BinaryAUROC, BinaryF1Score, BinaryAUPRC
from semilearn.nets import WNet
import higher
import copy
# from semilearn.core.utils import send_model_cuda

class MetaGradientHook(Hook):
    def __init__(self):
        super().__init__()

#     @torch.no_grad()
    def cal_meta_gradient(self,idx,lb_data, lb_label, unl_data_w, unl_data_s,unl_label,y_true,algorithm):
        algorithm.optimizer.zero_grad()
        # algorithm.optimizer_wnet.zero_grad()
        with higher.innerloop_ctx(algorithm.model, algorithm.optimizer) as (meta_model, meta_opt):
#                 for param_group in meta_opt.param_groups:
#                     param_group['lr'] = param_group['lr'] 

                # 1. update model with training data
                b=len(lb_data)
                eps = torch.tensor([1/b]*b,requires_grad=True).cuda()
                # eps = torch.zeros(len(lb_data), requires_grad=True).cuda()
                outputs = meta_model(lb_data)
                logits = outputs['logits']
                dens_feature = outputs['feat']
#                 tsa = algorithm.TSA(algorithm.tsa_schedule, algorithm.it, algorithm.num_train_iter, algorithm.num_classes)  # Training Signal Annealing
#                 sup_mask = torch.max(torch.softmax(logits, dim=-1), dim=-1)[0].le(tsa).float().detach()
#                 meta_loss = (ce_loss(logits,lb_label , reduction='none') * sup_mask)
                k=min(len(logits),algorithm.args.lid_k)

                lid_feat_l2 = lid(dens_feature, k=k, distmetric = 'l2')
                
                            
                meta_loss = ce_loss(logits, lb_label,reduction='none')
                # meta_model.zero_grad()
                meta_train_loss = torch.sum(eps * meta_loss)
                meta_opt.step(meta_train_loss)
                

                # 2. compute grads of eps on validation set
                if algorithm.args.weak_only:
                    outputs_u = meta_model(unl_data_w)
                    logits_u = outputs_u['logits']
                    feature_u = outputs_u['feat'] 
                    feat_u_w = feature_u
                    
                else:
                    inputs_u = torch.cat((unl_data_w, unl_data_s))
                    unl_label = torch.cat((unl_label,unl_label))
                    outputs_u = meta_model(inputs_u)
                    num_splits=2
                    logits_u = outputs_u['logits']
                    feature_u = outputs_u['feat']
                    
                    split_size = len(logits_u) // num_splits
                    logits_split = torch.split(logits_u, split_size)
                    feat_split = torch.split(feature_u, split_size)
                    logits_x_ulb_w, logits_x_ulb_s =logits_split[0],logits_split[1]
                    feat_u_w, feat_u_s = feat_split[0],feat_split[1]
                    probs = [F.softmax(logits, dim=1) for logits in logits_split]
                    logp_mixture = torch.clamp(torch.stack(probs).mean(axis=0), 1e-7, 1).log()
                    # JSD = self.alpha * sum([F.kl_div(logp_mixture, p_split, reduction='batchmean') for p_split in probs]) / len(probs)
                    
                feature_u = outputs_u['feat']
                

                ######## k=40
                k = 20
                # if algorithm.args.lid_est =='mle':
                lid_feat_l2 = lid(feat_u_w, k=k, distmetric = 'l2')
                # lid_feat_l2 =TwoNN(feat_u_w)
                # else:
                #     lid_feat_l2 = TwoNN(feature_u)
                unl_ce_loss = ce_loss(logits_u, unl_label,reduction='none')
                std_loss, cov_loss = variance_loss(feature_u) 
                
                mask = algorithm.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_w)
                pseudo_label = algorithm.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                                   logits=logits_x_ulb_w,
                                                   use_hard_label=False,
                                                   T=algorithm.T)
                
                unsup_loss = consistency_loss(logits_x_ulb_s,
                                              pseudo_label,
                                              'ce',
                                              mask=mask)                                

                
                y_val = torch.max(logits_u, dim=-1)[1].cpu().tolist()
                # print('Prediction:',torch.max(logits_u, dim=-1)[1].cpu().unique(),'True:',unl_label.cpu().unique())

                acc_val = accuracy_score(unl_label.cpu().tolist(), y_val)
                
                # th = algorithm.args.threshold
                
                # th = None if th == 'None' else th
                # # th = algorithm.args.s_it
                # # total_loss = manifold_raw(logits_u,th,kernel=False)
                # # entropy_intra_p, entropy_inter_p = entropy_p(feature_u,th)
                # # print(entropy_intra_p.item() , entropy_inter_p.item())
                # # total_loss = entropy_intra_p -entropy_inter_p
                # # total_loss = unl_ce_loss.mean()

                # if th is not None:
                #     if algorithm.args.meta_goal == 'feat':
                #         total_loss = manifold_loss_hard(feature_u,logits_u,th,kernel=False)+1
                #     elif algorithm.args.meta_goal == 'logits':
                #         total_loss = manifold_loss_hard(logits_u,logits_u,th,kernel=False)+1
                #     elif algorithm.args.meta_goal == 'feat_exp':
                #         total_loss = manifold_loss_hard_exp(feature_u,logits_u,th,kernel=False)+1
                #     elif algorithm.args.meta_goal == 'logits_exp':
                #         total_loss = manifold_loss_hard_exp(logits_u,logits_u,th,kernel=False)+1
                #     elif algorithm.args.meta_goal == 'manifold_raw':
                #         total_loss = manifold_raw(logits_u,th,kernel=False)+1
                #     elif algorithm.args.meta_goal == 'entr':
                #         entropy_intra_p, entropy_inter_p = entropy_p(logits_u,logits_u,th)
                #         total_loss = entropy_intra_p - entropy_inter_p
                #     elif algorithm.args.meta_goal == 'ce':
                #         total_loss = unl_ce_loss.mean()
                # else:
                if algorithm.args.meta_goal == 'feat':
                    total_loss = manifold_loss_soft(feature_u,logits_u,kernel=False)+1
                elif algorithm.args.meta_goal == 'logits':
                    total_loss = manifold_loss_soft(logits_u,logits_u,kernel=False)+1     
                elif algorithm.args.meta_goal == 'feat_exp':
                    total_loss = manifold_loss_soft_exp(feature_u,logits_u,contrast_th=algorithm.args.threshold)
                elif algorithm.args.meta_goal == 'feat_expr':
                    total_loss = manifold_loss_soft_expr(feature_u,logits_u,contrast_th=algorithm.args.threshold)
        
                elif algorithm.args.meta_goal == 'feat_expN':
                    total_loss = manifold_loss_soft_exp_(feature_u,logits_u,contrast_th=algorithm.args.threshold)        
                elif algorithm.args.meta_goal == 'feat_expNr':
                    total_loss = manifold_loss_soft_expr_(feature_u,logits_u,contrast_th=algorithm.args.threshold)        
                    
                elif algorithm.args.meta_goal == 'feat_expno1':
                    total_loss = manifold_loss_soft_exp_no1(feature_u,logits_u,contrast_th=algorithm.args.threshold)
                elif algorithm.args.meta_goal == 'feat_expno1r':
                    total_loss = manifold_loss_soft_exp_no1r(feature_u,logits_u,contrast_th=algorithm.args.threshold)
                elif algorithm.args.meta_goal == 'feat_expno1N':
                    total_loss = manifold_loss_soft_exp_no1_(feature_u,logits_u,contrast_th=algorithm.args.threshold)
                    
                elif algorithm.args.meta_goal == 'feat_expno1Nr':
                    total_loss = manifold_loss_soft_exp_no1r_(feature_u,logits_u,contrast_th=algorithm.args.threshold)

                elif algorithm.args.meta_goal == 'logits_exp':
                    total_loss = manifold_loss_soft_exp(logits_u,logits_u,kernel=False)+1

                elif algorithm.args.meta_goal == 'feat_exp_two':
                    total_loss = manifold_loss_soft_exp_two(feature_u,logits_u,kernel=False)
                elif algorithm.args.meta_goal == 'ce':
                    total_loss = unl_ce_loss.mean()
                elif algorithm.args.meta_goal =='JSD':
                    total_loss = JSD(feature_u,logits_u)                       
                elif algorithm.args.meta_goal =='JSDI':
                    total_loss = 12 * sum([F.kl_div(logp_mixture, p_split, reduction='batchmean') for p_split in probs]) / len(probs)
                elif algorithm.args.meta_goal == 'raw_u':
                    total_loss = unsup_loss
                elif algorithm.args.meta_goal == 'lid':
                    total_loss = lid_feat_l2
                elif algorithm.args.meta_goal == 'feat_exp_ws_':
                    total_loss = manifold_loss_soft_exp_ws_(feat_u_w,feat_u_s,logits_x_ulb_w,logits_x_ulb_s,algorithm.args.threshold)
                elif algorithm.args.meta_goal == 'feat_exp_ws':
                    total_loss = manifold_loss_soft_exp_ws(feat_u_w,feat_u_s,logits_x_ulb_w,logits_x_ulb_s,algorithm.args.threshold)
                elif algorithm.args.meta_goal == 'dfeat_expno1':
                    total_loss = dmanifold_loss_soft_exp_no1(feature_u,logits_u,contrast_th=algorithm.args.threshold)

                elif algorithm.args.meta_goal == 'dfeat_expno1N':
                    total_loss = dmanifold_loss_soft_exp_no1_(feature_u,logits_u,contrast_th=algorithm.args.threshold)
                else:
                    total_loss = manifold_loss_soft_exp_no1_(feature_u,logits_u,contrast_th=algorithm.args.threshold)
        
                # elif algorithm.args.meta_goal == 'graph_sim_dct':
                #     total_loss, matrics = graph_sim_dct(feature_u,logits_u,kernel=False)
                # elif algorithm.args.meta_goal == 'graph_sim_dct_n':
                #     total_loss, matrics = graph_sim_dct_n(feature_u,logits_u,kernel=False)
                # elif algorithm.args.meta_goal == "graph_sim_dct_ce":
                #     total_loss = graph_sim_dct_ce(feature_u,logits_u,kernel=False)
                # elif algorithm.args.meta_goal =='graph_sim_dct_ce_n':
                #     total_loss = graph_sim_dct_ce_n(feature_u,logits_u,contrast_th=algorithm.args.threshold)
                # elif algorithm.args.meta_goal =='graph_sim_dct_ce_nr':
                #     total_loss = graph_sim_dct_ce_nr(feature_u,logits_u,contrast_th=algorithm.args.threshold)
                # total_loss = lid_feat_l2
                # w_feat = - 1*torch.autograd.grad(total_loss, eps, only_inputs=True, retain_graph=True)[0].detach()    #l2
                if total_loss!=0:
                    
                    w_feat = eps.detach()-1*torch.autograd.grad(total_loss, eps,only_inputs=True,retain_graph=True)[0].detach()    #l2         

                else:
                    w_feat = torch.zeros(len(eps)).cuda()
                 
                # total_loss = manifold_loss(logits_u,th)+1
                # w_feat = -1*torch.autograd.grad(total_loss, eps,only_inputs=True,retain_graph=True)[0].detach()    #l2
        torch.set_warn_always(False)
        Auroc_f = BinaryAUROC()
        # F1_f = BinaryF1Score()
        Auprc_f = BinaryAUPRC()
        identified= torch.clamp(w_feat,min=0)
        identified[identified!=0] = 1
        identified = 1-identified # 1 means identified noisy label
                        
        mislabeled_ = (y_true!=lb_label).long().cpu() # True (1) means mislabeled sample

        auprc = manifold_loss_soft_exp(feature_u,logits_u,contrast_th=algorithm.args.threshold)
        auc = manifold_loss_soft_expr(feature_u,logits_u,contrast_th=algorithm.args.threshold)
        f1 = JSD(feature_u,logits_u)
        # sample_idx is the idx of labeled training samples
        if algorithm.args.meta_goal == 'graph_sim_dct' or algorithm.args.meta_goal == 'graph_sim_dct_n':
            weight_log ={'iteration':algorithm.it,'sample_idx': idx.cpu(),'labi_loss':meta_loss.cpu().detach(),\
                         # 'labi_lid_feat_l2':train_lid_feat_l2.cpu().detach().item(),\
                         'labi_lid_feat_l2': lb_label.cpu(),\
                         'w_lid_feat_l2':w_feat.cpu(),\
                        'lid_feat_l2':lid_feat_l2.item(),\
                         'total_meta_loss':total_loss.item(),\
                         'unl_ce_loss':unl_ce_loss.mean().item(),\
                        'feat_std': std_loss.item(),\
                        'cov_loss': cov_loss.item(),\
                         'auc': auc.item(),\
                         'f1': f1.item(),\
                         'auprc':[matrics,unl_label.cpu().detach()]\
                        }
        else:
            weight_log ={'iteration':algorithm.it,'sample_idx': idx.cpu(),'labi_loss':meta_loss.cpu().detach(),\
             # 'labi_lid_feat_l2':train_lid_feat_l2.cpu().detach().item(),\
             'labi_lid_feat_l2': lb_label.cpu(),\
             'w_lid_feat_l2':w_feat.cpu(),\
            'lid_feat_l2':lid_feat_l2.item(),\
             'total_meta_loss':total_loss.item(),\
             'unl_ce_loss':unl_ce_loss.mean().item(),\
            'feat_std': std_loss.item(),\
            'cov_loss': cov_loss.item(),\
             'auc': auc.item(),\
             'f1': f1.item(),\
             'auprc':auprc.item()\
            }
 
            
                
        tb_meta={'lid_80':{'lid_feat_l2':lid_feat_l2.item(),
                        'unl_ce_loss':unl_ce_loss.mean().item(),\
                }}                


              
        return tb_meta, weight_log, w_feat, acc_val
