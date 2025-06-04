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
import numpy as np
# from semilearn.core.utils import send_model_cuda
torch.autograd.set_detect_anomaly(True)

class MetaLossNetHook(Hook):
    def __init__(self):
        super().__init__()

#     @torch.no_grad()
    def cal_meta_gradient(self,idx,lb_data, lb_label,y_true,algorithm):
        algorithm.optimizer.zero_grad()
        try:
            data_meta=next(algorithm.meta_loader)
        except:
            algorithm.meta_loader = iter(algorithm.loader_dict['meta'])
            data_meta = next(algorithm.meta_loader)
        
        meta_dict = algorithm.process_batch(**data_meta)
        if 'x_lb' in meta_dict.keys():
            meta_data_w = meta_dict['x_lb']
            meta_label = meta_dict['y_lb']
            meta_label_true = meta_dict['y_true'] 
            algorithm.args.weak_only = True
            # print('clean:',meta_label_true,'\n','nsoisy:',meta_label)
        elif 'x_ulb_w' in meta_dict.keys():
            meta_data_w = meta_dict['x_ulb_w']
            meta_label_true = meta_dict['y_ulb_true'] 

            if 'x_ulb_s' in meta_dict.keys():
                meta_data_s = meta_dict['x_ulb_s']
            elif 'x_ulb_s_0' in meta_dict.keys():
                meta_data_s = meta_dict['x_ulb_s_0']
            else:
                algorithm.args.weak_only = True
                
            
            
        # algorithm.optimizer_wnet.zero_grad()
        with higher.innerloop_ctx(algorithm.model, algorithm.optimizer) as (meta_model, meta_opt):                
            # Freeze the encoder layers
            # if algorithm.args.frozen:
                # for param in meta_model.conv1.parameters():
                #     param.requires_grad = False
                # for param in meta_model.block1.parameters():
                #     param.requires_grad = False
                # for param in meta_model.block2.parameters():
                #     param.requires_grad = False
                # for param in meta_model.block3.parameters():
                #     param.requires_grad = False
                # for param in meta_model.parameters():
                #     param.requires_grad = False
                # for param in meta_model.fc.parameters():
                #     param.requires_grad = True                
            # if algorithm.args.meta_goal == 'feat_expno1':
            #     for param in meta_model.fc.parameters():
            #         param.requires_grad = False
                    
            outputs = meta_model(lb_data)
            logits = outputs['logits']
            dens_feature = outputs['feat']

            k=min(len(logits),algorithm.args.lid_k)

            lid_feat_l2 = lid(dens_feature, k=k, distmetric = 'l2')


            virtual_loss = ce_loss(logits, lb_label,reduction='none')

            # meta loss
            cost_v = torch.reshape(virtual_loss , (len(virtual_loss), 1))

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
                outputs_u = meta_model(meta_data_w)
                logits_u = outputs_u['logits']
                feature_u = outputs_u['feat'] 
                feat_u_w = feature_u
                
            else:
                inputs_u = torch.cat((meta_data_w, meta_data_s))
                meta_label_true = torch.cat((meta_label_true,meta_label_true))
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
            k = 10
            lid_feat_l2 = lid(feature_u, k=k, distmetric = 'l2')
            # else:
            #     lid_feat_l2 = TwoNN(feature_u)
            unl_ce_loss = ce_loss(logits_u, meta_label_true,reduction='none')
            std_loss, cov_loss = variance_loss(feature_u) 
            # contra_intra = entropy_intra(logits_u, logits_u)
            


            y_val=torch.max(logits_u, dim=-1)[1]
            acc_val = accuracy_score(meta_label_true.cpu().tolist(), y_val.cpu().tolist())
            


            if algorithm.args.meta_goal == 'feat':
                total_loss = manifold_loss_soft(feature_u,logits_u,kernel=False)+1
            elif algorithm.args.meta_goal == 'logits':
                total_loss = manifold_loss_soft(logits_u,logits_u,kernel=False)+1     
            elif algorithm.args.meta_goal == 'feat_exp':
                total_loss = manifold_loss_soft_exp(feature_u,logits_u,contrast_th=algorithm.args.threshold)
            elif algorithm.args.meta_goal == 'feat_expno1N': ##
                total_loss = manifold_loss_soft_exp_no1_(feature_u,logits_u,contrast_th=algorithm.args.threshold)
            elif algorithm.args.meta_goal == 'feat_expno1': ##
                total_loss = manifold_loss_soft_exp_no1(feature_u,logits_u,contrast_th=algorithm.args.threshold)
                
            elif algorithm.args.meta_goal == 'hfeat_expno1N': ##
                total_loss = manifold_loss_soft_exp_no1_hard(feature_u,logits_u,contrast_th=algorithm.args.threshold)
            elif algorithm.args.meta_goal == 'sfeat_expno1N': ##
                total_loss = manifold_loss_soft_exp_no1_soft(feature_u,logits_u,contrast_th=algorithm.args.threshold)
                
            elif algorithm.args.meta_goal == 'feat_expno1Nr':
                total_loss = manifold_loss_soft_exp_no1r_(feature_u,logits_u,contrast_th=algorithm.args.threshold)

            elif algorithm.args.meta_goal == 'logits_exp':
                total_loss = manifold_loss_soft_exp(logits_u,logits_u,kernel=False)+1

            elif algorithm.args.meta_goal == 'feat_exp_two':
                total_loss = manifold_loss_soft_exp_two(feature_u,logits_u,kernel=False)
            elif algorithm.args.meta_goal == 'cer':
                total_loss = ce_loss(logits_u, meta_label_true,reduction='mean')
            elif algorithm.args.meta_goal == 'cen':
                total_loss = ce_loss(logits_u, meta_label,reduction='mean')
            elif algorithm.args.meta_goal == 'mae':
                total_loss = mae_loss(logits_u, meta_label,reduction='mean')
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

            Continue = True
        
            # s_list_metrics = []
            
            # if algorithm.log['total_meta_loss']==[]:
            #     list_metrics = []
            # else:
            #     list_metrics = algorithm.log['total_meta_loss']
            # list_metrics.append(total_loss.item())
            # p_l=0
            # smoothing_alpha=0.999
            # for i in range(len(list_metrics)):
            #     p_l = smoothing_alpha *p_l + (1 - smoothing_alpha)* list_metrics[i]
            #     s_list_metrics.append(p_l/(1 - smoothing_alpha**(i+1)))
            #     diff=10000
            #     if i>3000:
            #         diff=np.mean(s_list_metrics[-5000:-1])-s_list_metrics[-1]
            #     if diff<0:
            #         Continue=False   

            ## update wnet
            if Continue:
                algorithm.optimizer_wnet.zero_grad()
                total_loss.backward()
                algorithm.optimizer_wnet.step()
                # algorithm.wnet_scheduler.step()




        # print(total_loss.item())
        # convert onehot to label
        mislabeled_ = (y_true!=lb_label).long().cpu().numpy() # True means mislabeled sample
        # sample_idx is the idx of labeled training samples
        weight_log ={'iteration':algorithm.it,'sample_idx': idx.cpu(),'labi_loss':virtual_loss.cpu().detach(),\
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
