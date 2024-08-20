# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import copy 

from semilearn.core.hooks import Hook
from semilearn.algorithms.utils.loss import *

from sklearn.metrics import confusion_matrix,roc_auc_score,f1_score,accuracy_score, balanced_accuracy_score
from semilearn.nets import WNet, MLP
import higher
import copy
# from semilearn.core.utils import send_model_cuda

class MetaLossesNetHook(Hook):
    def __init__(self):
        super().__init__()

#     @torch.no_grad()
    def cal_meta_gradient(self,idx,lb_data, lb_label, unl_data_w, unl_data_s,unl_label,y_true,algorithm):
        algorithm.optimizer.zero_grad()
        # algorithm.optimizer_wnet.zero_grad()
        with higher.innerloop_ctx(algorithm.model, algorithm.optimizer) as (meta_model, meta_opt):                

            b=len(lb_data)
            eps = torch.tensor([1/b]*b,requires_grad=True).cuda()
            # eps = torch.zeros(len(lb_data), requires_grad=True).cuda()
            outputs = meta_model(lb_data)
            logits = outputs['logits']
            dens_feature = outputs['feat']

            k=min(len(logits),algorithm.args.lid_k)

            lid_feat_l2 = lid(dens_feature, k=k, distmetric = 'l2')


            meta_ce_loss = ce_loss(logits, lb_label,reduction='none')
            meta_gce_loss = gce_loss(logits, lb_label, q=0.7,reduction='none')
            meta_consistency_loss = consistency_loss()
            # meta loss
            cost_v = torch.reshape(meta_loss , (len(meta_loss), 1))
            # meta_model.zero_grad()

            # if norm!=0:
            #     meta_train_loss = torch.sum(eps * meta_loss)/norm
            # else:
            #     meta_train_loss = torch.sum(eps * meta_loss)
            v_lambda = algorithm.wnet(cost_v.data)
            # norm_c = torch.sum(v_lambda)
    
            # if norm_c != 0:
            #     v_lambda = v_lambda / norm_c
            # else:
            #     v_lambda = v_lambda

            # meta_train_loss = torch.sum(cost_v * v_lambda)
            meta_train_loss = torch.mean(cost_v * v_lambda)

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
            k = 10
            lid_feat_l2 = lid(feature_u, k=k, distmetric = 'l2')
            # else:
            #     lid_feat_l2 = TwoNN(feature_u)
            unl_ce_loss = ce_loss(logits_u, unl_label,reduction='none')
            std_loss, cov_loss = variance_loss(feature_u) 
            # contra_intra = entropy_intra(logits_u, logits_u)
            


            y_val=torch.max(logits_u, dim=-1)[1]
            acc_val = accuracy_score(unl_label.cpu().tolist(), y_val.cpu().tolist())
            


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
                total_loss = ce_loss(logits, y_true,reduction='mean')
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

    

                
            ## update wnet
            algorithm.optimizer_wnet.zero_grad()
            total_loss.backward()
            algorithm.optimizer_wnet.step()
            algorithm.optimizer.zero_grad()




        # print(total_loss.item())
        # convert onehot to label
        mislabeled_ = (y_true!=lb_label).long().cpu().numpy() # True means mislabeled sample
        # sample_idx is the idx of labeled training samples
        weight_log ={'iteration':algorithm.it,'sample_idx': idx.cpu(),'labi_loss':meta_loss.cpu().detach(),\
                     'labi_lid_feat_l2':lb_label.cpu(),\
                    'lid_feat_l2':lid_feat_l2.item(),\
                     'total_meta_loss':total_loss.item(),\
                     'unl_ce_loss':unl_ce_loss.mean().item(),\
                    'feat_std': std_loss.item(),\
                    'cov_loss': cov_loss.item(),\
                    }




        tb_meta={'lid_80':{'lid_feat_l2':lid_feat_l2.item(),
                        'unl_ce_loss':unl_ce_loss.mean().item(),\
                }}                


    


        
              
        return tb_meta, weight_log, acc_val
