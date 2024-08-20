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

class MetaNetHook(Hook):
    def __init__(self):
        super().__init__()

#     @torch.no_grad()
    def cal_meta_gradient(self,idx,lb_data, lb_label, unl_data_w, unl_data_s,unl_label,y_true,algorithm):
        algorithm.optimizer.zero_grad()
        # algorithm.optimizer_wnet.zero_grad()
        with higher.innerloop_ctx(algorithm.model, algorithm.optimizer) as (meta_model, meta_opt):                
            outputs = meta_model(lb_data)
            logits = outputs['logits']
            dens_feature = outputs['feat']

             

#                 tsa = algorithm.TSA(algorithm.tsa_schedule, algorithm.it, algorithm.num_train_iter, algorithm.num_classes)  # Training Signal Annealing
#                 sup_mask = torch.max(torch.softmax(logits, dim=-1), dim=-1)[0].le(tsa).float().detach()
#                 meta_loss = (ce_loss(logits,lb_label , reduction='none') * sup_mask)
            k=min(len(logits),algorithm.args.lid_k)

            train_lid_feat_l2 = lid(dens_feature, k=k, distmetric = 'l2')


            meta_loss = ce_loss(logits, lb_label,reduction='none')
            
            w = algorithm.wnet(logits.softmax(1))
            w = w.squeeze()
            # w = (w-w.min())/(w.max()-w.min()) # or weighted mean, don't have to make the sum equal to 1
            # norm = torch.sum(w)
            # if norm!=0:
            #     w = w/norm
            meta_train_loss = torch.sum(meta_loss * w/len(w))
            # meta loss
            # cost_v = torch.reshape(meta_loss , (len(meta_loss), 1))
            # eps = algorithm.wnet(cost_v.data)
            # meta_model.zero_grad()

            # if norm!=0:
            #     meta_train_loss = torch.sum(eps * meta_loss)/norm
            # else:
            #     meta_train_loss = torch.sum(eps * meta_loss)
            # v_lambda = algorithm.wnet(cost_v.data)
            # norm_c = torch.sum(v_lambda)
#             if norm_c != 0:
#                 v_lambda_norm = v_lambda / norm_c
#             else:
#                 v_lambda_norm = v_lambda

            # meta_train_loss = torch.sum(cost_v * v_lambda_norm)
            
            meta_opt.step(meta_train_loss)


            # 2. compute grads of eps on validation set i.e., compute upper level objective
            if algorithm.args.weak_only:
                outputs_u = meta_model(unl_data_w)

            else:
                inputs_u = torch.cat((unl_data_w, unl_data_s))
                unl_label = torch.cat((unl_label,unl_label))
                outputs_u = meta_model(inputs_u)

            logits_u = outputs_u['logits']
            feature_u = outputs_u['feat']

    #         outs_x_lb = self.model(x_lb) 
    #         logits_x_lb = outs_x_lb['logits']
    #         outs_x_ulb_s = self.model(x_ulb_s)
    #         logits_x_ulb_s = outs_x_ulb_s['logits']
    #         with torch.no_grad():
    #             outs_x_ulb_w = self.model(x_ulb_w)
    #             logits_x_ulb_w = outs_x_ulb_w['logits']
        # compute mask
        # mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_w)                

            ######## k=40
            k = algorithm.args.lid_k
            
            lid_feat_l2 = lid(feature_u, k=k, distmetric = 'l2')
            
            # lid_feat_l2 = TwoNN(feature_u)
            unl_ce_loss = ce_loss(logits_u, unl_label,reduction='none')
            std_loss, cov_loss = variance_loss(feature_u) 
            # contra_intra = entropy_intra(logits_u, logits_u)
            


            y_val=torch.max(logits_u, dim=-1)[1]
            acc_val = accuracy_score(unl_label.cpu().tolist(), y_val.cpu().tolist())
            
            
            th = algorithm.args.threshold
            th = None if th == 'None' else th
            # entr_inter, entr_intra = entropy(logits_u, logits_u,th)
            # total_loss = torch.exp(-entr_intra) + torch.exp(entr_inter)
             
            # total_loss = manifold_loss(logits_u,th)+1
            # total_loss = manifold_raw(logits_u,th,kernel=False)
            
            if th is not None:
                if algorithm.args.meta_goal == 'feat':
                    total_loss = manifold_loss_hard(feature_u,logits_u,th,kernel=False)+1
                elif algorithm.args.meta_goal == 'logits':
                    total_loss = manifold_loss_hard(logits_u,logits_u,th,kernel=False)+1
                elif algorithm.args.meta_goal == 'feat_exp':
                    total_loss = manifold_loss_hard_exp(feature_u,logits_u,th,kernel=False)+1
                elif algorithm.args.meta_goal == 'logits_exp':
                    total_loss = manifold_loss_hard_exp(logits_u,logits_u,th,kernel=False)+1
                elif algorithm.args.meta_goal == 'manifold_raw':
                    total_loss = manifold_raw(logits_u,th,kernel=False)+1
                elif algorithm.args.meta_goal == 'entr':
                    entropy_intra_p, entropy_inter_p = entropy_p(feature_u,logits_u,th)
                    total_loss = entropy_intra_p - entropy_inter_p
                elif algorithm.args.meta_goal == 'ce':
                    total_loss = unl_ce_loss.mean()
            else:
                if algorithm.args.meta_goal == 'feat':
                    total_loss = manifold_loss_soft(feature_u,logits_u,kernel=False)+1
                elif algorithm.args.meta_goal == 'logits':
                    total_loss = manifold_loss_soft(logits_u,logits_u,kernel=False)+1     
                elif algorithm.args.meta_goal == 'feat_exp':
                    # print(algorithm.it)
                    if algorithm.it<800:
                        total_loss = manifold_loss_soft_exp(feature_u,logits_u,kernel=False)+1
                    else:
                        total_loss = -manifold_loss_soft_exp(feature_u,logits_u,kernel=False)+1
                elif algorithm.args.meta_goal == 'logits_exp':
                    total_loss = manifold_loss_soft_exp(logits_u,logits_u,kernel=False)+1
                elif algorithm.args.meta_goal == 'feat_exp_two':
                    total_loss = manifold_loss_soft_exp_two(feature_u,logits_u,kernel=False)
                elif algorithm.args.meta_goal == 'ce':
                    total_loss = unl_ce_loss.mean()
                elif algorithm.args.meta_goal =='feat_exp_r':
                    total_loss = manifold_loss_soft_exp_r(feature_u,logits_u,kernel=False)
                elif algorithm.args.meta_goal =='feat_exp_r_detach':
                    total_loss = manifold_loss_soft_exp_r_detach(feature_u,logits_u,kernel=False)
                elif algorithm.args.meta_goal =='feat_exp_detach':
                    total_loss = manifold_loss_soft_exp_detach(feature_u,logits_u,kernel=False)
                elif algorithm.args.meta_goal == 'feat_exp_detach2': 
                    total_loss = manifold_loss_soft_exp_detach2(feature_u,logits_u,kernel=False)
                elif algorithm.args.meta_goal =='JSD':
                    total_loss = JSD(feature_u,logits_u)
                else:
                    total_loss = manifold_loss_soft_exp(feature_u,logits_u,kernel=False)+1

            # entropy_intra_p, entropy_inter_p = entropy_p(logits_u,th)
            # total_loss = entropy_intra_p - entropy_inter_p
            # print('entropy_intra_p:',entropy_intra_p.item(),'entropy_inter_p:',entropy_inter_p.item())
            # total_loss = unl_ce_loss.mean()

                
            ## update wnet
            # print(total_loss)
        algorithm.optimizer_wnet.zero_grad()
        total_loss.backward()
        algorithm.optimizer_wnet.step()





        # print(total_loss.item())
        # convert onehot to label
        torch.set_warn_always(False)

        auprc = manifold_loss_soft_exp(feature_u,logits_u,kernel=False)
        auc = manifold_loss_soft_exp_r(feature_u,logits_u,kernel=False)
        f1 = JSD(feature_u,logits_u)


        
        
        # sample_idx is the idx of labeled training samples
        weight_log ={'iteration':algorithm.it,'sample_idx': idx.cpu(),'labi_loss':meta_loss.cpu().detach(),\
                     # 'labi_lid_feat_l2':train_lid_feat_l2.cpu().detach().item(),\
                     'labi_lid_feat_l2': lb_label.cpu(),\
                    'lid_feat_l2':lid_feat_l2.item(),\
                     'total_meta_loss':total_loss.item(),\
                     'unl_ce_loss':unl_ce_loss.mean().item(),\
                    'feat_std': std_loss.item(),\
                    'cov_loss': cov_loss.item(),\
                     'auc': auc.item(),\
                     'f1':f1.item(),\
                     'auprc':auprc.item()\
                    }






        tb_meta={'lid_80':{'lid_feat_l2':lid_feat_l2.item(),
                        'unl_ce_loss':unl_ce_loss.mean().item(),\
                }}                


    


        
                      
        return tb_meta, weight_log, acc_val
 
            
        
              
           