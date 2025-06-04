# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import copy 
import torch
import math
from semilearn.core import AlgorithmBase
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook, MetaGradientHook
from semilearn.algorithms.utils import ce_loss, consistency_loss, SSL_Argument
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
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']

            tsa = self.TSA(self.tsa_schedule, self.it, self.num_train_iter, self.num_classes)  # Training Signal Annealing
            
            sup_mask = torch.max(torch.softmax(logits_x_lb, dim=-1), dim=-1)[0].le(tsa).float().detach()
            print(sup_mask.sum())
#             sup_loss = (ce_loss(logits_x_lb, y_lb, reduction='none') * sup_mask).mean()
#             sup_loss_true = (ce_loss(logits_x_lb, y_true, reduction='none') * sup_mask).mean()

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_w)
            
            # compute weght for labeled sample and calculate the identifying performance in terms of F1
#             for param_group in self.optimizer.param_groups:
#                 lr_inner = param_group['lr']
                # model,lb_data, lb_label, unl_data,unl_label,k,lr,distmetric,y_true\
            meta_hook = MetaGradientHook()

            tb_meta, weight_log, w_eps,acc_val = meta_hook.cal_meta_gradient(idx=idx_lb,lb_data=x_lb, lb_label=y_lb, unl_data_w=x_e_w,\
                                                              unl_data_s=x_e_s, unl_label=y_e_true,\
                                                              y_true=y_true, algorithm=self)
#             tb_meta = self.call_hook("cal_meta_gradient","MetaGradientHook", x_lb,y_lb, x_ulb_w2, y_ulb_true2, 40,'l2', y_true, self.model, self.optimizer)
            
            # Line 11 computing and normalizing the weights
            if self.args.normalize_w == 'zero':
                total_meta = torch.maximum(torch.zeros(len(w_eps)).cuda(), w_eps)
                if torch.sum(total_meta)==0:
                    total_meta = torch.zeros(len(w_eps)).cuda()
                else:
                    total_meta /= torch.sum(total_meta)  # or weighted mean, don't have to make the sum equal to 1
            else:
                total_meta = (w_eps-w_eps.min())/(w_eps.max()-w_eps.min()) # or weighted mean, don't have to make the sum equal to 1
                total_meta /= torch.sum(total_meta) 
                
                
            if self.args.meta_updating and self.it>self.args.s_it:
#                 print('\n','meta',self.it, '\n')
                sup_loss = torch.sum(ce_loss(logits_x_lb, y_lb, reduction='none') * sup_mask*total_meta)
                sup_loss_true = (ce_loss(logits_x_lb, y_true, reduction='none') * sup_mask).mean()
                sup_loss_true_weighted = (ce_loss(logits_x_lb, y_true, reduction='none') * sup_mask*total_meta).sum().item()
                sup_loss_raw = (ce_loss(logits_x_lb, y_lb, reduction='none') * sup_mask).mean().item()
            else:
                
                sup_loss = (ce_loss(logits_x_lb, y_lb, reduction='none') * sup_mask).mean()
                sup_loss_true = (ce_loss(logits_x_lb, y_true, reduction='none') * sup_mask).mean()
                sup_loss_true_weighted = 0.0
                sup_loss_raw = sup_loss.item()
            
            
        
        
        ######################################
    
    
    
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=logits_x_ulb_w,
                                          use_hard_label=False,
                                          T=self.T)

            unsup_loss = consistency_loss(logits_x_ulb_s,
                                          pseudo_label,
                                          'ce',
                                          mask=mask)
#             if unsup_loss.item()>0:
#                 self.lambda_u = torch.ceil(sup_loss/unsup_loss).item()
#                 print(self.lambda_u)
            total_loss = sup_loss + self.lambda_u * unsup_loss
            

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
        tb_dict['train/mask_ratio'] = mask.float().mean().item()
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
