# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import copy 
import torch
import math
from semilearn.core import AlgorithmBase
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook, MetaGradientHook
# from semilearn.algorithms.utils import ce_loss, consistency_loss, SSL_Argument, gradient_match, lid, TwoNN,manifold_loss_soft,manifold_loss_soft_exp
from semilearn.algorithms.utils import *
from sklearn.metrics import confusion_matrix,roc_auc_score,f1_score,accuracy_score, balanced_accuracy_score

torch.autograd.set_detect_anomaly(True)

class UDA(AlgorithmBase):
    """
    UDA algorithm (https://arxiv.org/abs/1904.12848).

    Args:
        - args (`argparse`):
            algorithm arguments
        - net_builder (`callable`):
            network loading function
        - tb_log (`TBLog`):
            tensorboard logger
        - logger (`logging.Logger`):
            logger to use
        - T (`float`):
            Temperature for pseudo-label sharpening
        - p_cutoff(`float`):
            Confidence threshold for generating pseudo-labels
        - hard_label (`bool`, *optional*, default to `False`):
            If True, targets have [Batch size] shape with int values. If False, the target is vector
        - tsa_schedule ('str'):
            TSA schedule to use
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # uda specificed arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, log=args.log, tsa_schedule=args.tsa_schedule)

    def init(self, T, p_cutoff, log,tsa_schedule='none'):
        self.T = T
        self.p_cutoff = p_cutoff
        self.tsa_schedule = tsa_schedule
        self.log=log
        
    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        self.register_hook(MetaGradientHook(), "MetaGradientHook")

        super().set_hooks()
#['idx_e', 'x_e_w', 'x_e_s', 'y_e_true']
    def train_step(self, idx_lb,x_lb, y_lb,y_true, idx_ulb, x_ulb_w, x_ulb_s, y_ulb_true, idx_e, x_e_w, x_e_s, y_e_true):
        num_lb = y_lb.shape[0]
        # inference and calculate sup/unsup losses
        with self.amp_cm():
            
            # compute weght for labeled sample and calculate the identifying performance in terms of F1
            if self.args.steplr and self.it==10000:
                for param_group in self.optimizer.param_groups:
                    param_group['lr']=param_group['lr']/10
                # model,lb_data, lb_label, unl_data,unl_label,k,lr,distmetric,y_true\
            meta_hook = MetaGradientHook()

            tb_meta, weight_log, w_eps,acc_val = meta_hook.cal_meta_gradient(idx=idx_lb,lb_data=x_lb, lb_label=y_lb, unl_data_w=x_e_w,\
                                                              unl_data_s=x_e_s, unl_label=y_e_true,\
                                                              y_true=y_true, algorithm=self)
#             tb_meta = self.call_hook("cal_meta_gradient","MetaGradientHook", x_lb,y_lb, x_ulb_w2, y_ulb_true2, 40,'l2', y_true, self.model, self.optimizer)
            
            # Line 11 computing and normalizing the weights
            if self.args.w_f == 'zero':
                total_meta = torch.clamp(w_eps, min=0)
                if torch.sum(total_meta)==0:
                    w = total_meta
                else:
                    w = total_meta/torch.sum(total_meta)  # or weighted mean, don't have to make the sum equal to 1
            elif self.args.w_f == 'min':
                total_meta = (w_eps-w_eps.min()) # or weighted mean, don't have to make the sum equal to 1
                w = total_meta/torch.sum(total_meta)  # or weighted mean, don't have to make the sum equal to 1
            elif self.args.w_f == 'uni':
                import copy
                n_samples=len(w_eps)
                threshold = torch.mean(w_eps)
                # print(threshold ,'------')
                # if threshold.item()> 0: #self.args.d_u
                #     threshold = threshold - (threshold-torch.min(w_eps))/5 #self.arg.tau
                DR = torch.sum(w_eps<threshold).item()/n_samples  #drop out ratio
                w = copy.deepcopy(w_eps) 
                idx_ts = []
                idx_bs = []
                for k in range(len(torch.unique(y_lb))):
                    idx= torch.where(y_lb==k)[0]
                    
                    n=int((1-DR)*len(idx))
                    idx_t  =torch.argsort(w_eps[idx])[-n:]
                    
                    idx_b = torch.argsort(w_eps[idx])[:(len(idx)-n)]
                    idx_ts.append(idx[idx_t])
                    idx_bs.append(idx[idx_b])
                idx_bs_ = torch.concat(idx_bs)
                idx_ts_ = torch.concat(idx_ts)
                
                w[idx_bs_] = torch.zeros(len(idx_bs_), requires_grad=False).cuda()
                w_eps_ = w_eps[idx_ts_]
                # total_meta = w_eps_-w_eps_.min() # or weighted mean, don't have to make the sum equal to 1
                w[idx_ts_] = torch.softmax(w_eps_,dim=-1)
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                feature_u_w, feature_u_s = outputs['feat'][num_lb:].chunk(2)
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_w)
            outs_x_lb = self.model(x_lb) 
            logits_x_lb = outs_x_lb['logits']
            
            tsa = self.TSA(self.tsa_schedule, self.it, self.num_train_iter, self.num_classes)  # Training Signal Annealing
            
            sup_mask = torch.max(torch.softmax(logits_x_lb, dim=-1), dim=-1)[0].le(tsa).float().detach()
            
#             sup_loss = (ce_loss(logits_x_lb, y_lb, reduction='none') * sup_mask).mean()
#             sup_loss_true = (ce_loss(logits_x_lb, y_true, reduction='none') * sup_mask).mean()

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_w)
            outs_x_ulb_w = self.model(x_ulb_w)
            logits_x_ulb_w = outs_x_ulb_w['logits']
            probs_x_ulb = torch.softmax(logits_x_ulb_w, dim=-1)
            max_probs, y = torch.max(probs_x_ulb, dim=-1)
            flag = torch.sum(max_probs.ge(0.7)).item()>0                    
            # if self.args.meta_updating and self.it>self.args.s_it:
            
            self.args.meta_goal = None if self.args.meta_goal == 'None' else self.args.meta_goal
            if self.args.meta_goal is not None and self.it>self.args.s_it:
            # if self.args.meta_goal is not None and torch.sum(w) >0:

                # print('meta! \n')
                sup_loss = torch.sum(ce_loss(logits_x_lb, y_lb, reduction='none') *w)
                sup_loss_true = ce_loss(logits_x_lb, y_true, reduction='mean')
                sup_loss_true_weighted = (ce_loss(logits_x_lb, y_true, reduction='none') *w).sum().item()
                sup_loss_raw = ce_loss(logits_x_lb, y_lb, reduction='mean').mean().item()
            else:    
                sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
                sup_loss_true = ce_loss(logits_x_lb, y_true, reduction='mean')
                sup_loss_true_weighted = 0.0
                sup_loss_raw = sup_loss.item()
            
            
        
        
        ######################################
    
    


            ######## k=40
            # k = 3
            # # if algorithm.args.lid_est =='mle':
            # # lid_feat_l2 = lid(feature_u_w, k=k, distmetric = 'l2')
            # lid_feat_l2 =TwoNN(feature_u_w)

            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=logits_x_ulb_w,
                                          use_hard_label=False,
                                          T=self.T)
            
            # if self.args.meta_goal == 'raw_u':
            unsup_loss = consistency_loss(logits_x_ulb_s,
                                          pseudo_label,
                                          'ce',
                                          mask=mask)
            # elif self.args.meta_goal == 'feat_exp':
            #     unsup_loss = manifold_loss_soft_exp(feature_u_w,logits_x_ulb_w,kernel=False)
            # if unsup_loss.item()>0:
            #     self.lambda_u = torch.ceil(sup_loss/unsup_loss).item()
            #     print(self.lambda_u)
            total_loss = sup_loss + self.lambda_u * unsup_loss
            # total_loss = sup_loss 

        y_pred_lb = torch.max(logits_x_lb, dim=-1)[1].cpu().tolist()
        acc_lb = accuracy_score(y_lb.cpu().tolist(), y_pred_lb)        
        acc_lb_true = accuracy_score(y_true.cpu().tolist(), y_pred_lb)   
        # parameter updates
        self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)
        
        tb_dict = {}
        tb_dict['train/acc_lb'] = acc_lb
        tb_dict['train/acc_lb_true'] = acc_lb_true
        
        tb_dict['train/sup_loss'] = sup_loss_raw
        tb_dict['train/sup_loss_weighted'] = sup_loss.item()
        tb_dict['train/sup_loss_true'] = sup_loss_true.item()
        tb_dict['train/sup_loss_true_weighted'] = sup_loss_true_weighted
        tb_dict['train_val_acc'] = acc_val
        tb_dict['train/unsup_loss'] = unsup_loss.item()
        tb_dict['train/total_loss'] = total_loss.item()
        # tb_dict['train/mask_ratio'] = mask.float().mean().item()
#         weight_log ={'iteration':self.it,'sample_idx': idx,'w_ce_loss':w_ce_loss , 'w_lid_logits':w_lid_logits, 'w_lid_feat':w_lid_feat}
        
        for key, value in weight_log.items():
            
            self.log[key].append(value)
        
        return tb_dict,tb_meta, self.log

    def TSA(self, schedule, cur_iter, total_iter, num_classes):
        training_progress = cur_iter / total_iter

        if schedule == 'linear':
            threshold = training_progress
        elif schedule == 'exp':
            scale = 5
            threshold = math.exp((training_progress - 1) * scale)
        elif schedule == 'log':
            scale = 5
            threshold = 1 - math.exp((-training_progress) * scale)
        elif schedule == 'none':
            return 1
        tsa = threshold * (1 - 1 / num_classes) + 1 / num_classes
        return tsa

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--tsa_schedule', str, 'none', 'TSA mode: none, linear, log, exp'),
            SSL_Argument('--T', float, 0.4, 'Temperature sharpening'),
            SSL_Argument('--p_cutoff', float, 0.8, 'confidencial masking threshold'),
            # SSL_Argument('--use_flex', str2bool, False),
        ]
