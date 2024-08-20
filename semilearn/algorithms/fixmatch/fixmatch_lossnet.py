

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook,MetaGradientHook,MetaLossNetHook
from semilearn.algorithms.utils import *
from sklearn.metrics import confusion_matrix,roc_auc_score,f1_score,accuracy_score, balanced_accuracy_score


class FixMatch_lossnet(AlgorithmBase):
    """
        FixMatch algorithm (https://arxiv.org/abs/2001.07685).

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
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # fixmatch specificed arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, log=args.log,hard_label=args.hard_label)
    
    def init(self, T, p_cutoff, log, hard_label=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.log=log
    
    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        self.register_hook(MetaLossNetHook(), "MetaLossNetHook")

        super().set_hooks()
        
    # def train_step(self, idx_lb,x_lb, y_lb,y_true, idx_ulb, x_ulb_w, x_ulb_s, y_ulb_true, idx_e, x_e_w, x_e_s, y_e_true):
    def train_step(self, idx_lb,x_lb, y_lb,y_true, idx_ulb, x_ulb_w, x_ulb_s, y_ulb_true):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            meta_hook = MetaLossNetHook()
            # tb_meta, weight_log, acc_val = meta_hook.cal_meta_gradient(idx=idx_lb,lb_data=x_lb, lb_label=y_lb, unl_data_w=x_e_w,\
            #                                                   unl_data_s=x_e_s, unl_label=y_e_true,\
            #                                                   y_true=y_true, algorithm=self)
            tb_meta, weight_log, acc_val = meta_hook.cal_meta_gradient(idx=idx_lb,lb_data=x_lb, lb_label=y_lb,\
                                                              y_true=y_true, algorithm=self)   
                 
                    
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                feat_u = outputs['feat'][num_lb:]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']

            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
            
            probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            # if distribution alignment hook is registered, call it 
            # this is implemented for imbalanced algorithm - CReST
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)

            unsup_loss = consistency_loss(logits_x_ulb_s,
                                          pseudo_label,
                                          'ce',
                                          mask=mask)
            
            cost_w= ce_loss(logits_x_lb, y_lb, reduction='none')
            cost_v=torch.reshape(cost_w, (len(cost_w), 1))
            with torch.no_grad():
                w = self.wnet(cost_v)

            weight_log['w_lid_feat_l2']= torch.squeeze(w).detach().cpu()

            
            self.args.meta_goal = None if self.args.meta_goal == 'None' else self.args.meta_goal
            # if self.args.meta_goal is not None:
            if self.args.meta_goal is not None:

                sup_loss = torch.mean(ce_loss(logits_x_lb, y_lb, reduction='none')*torch.squeeze(w))
                sup_loss_true = ce_loss(logits_x_lb, y_true, reduction='mean')
                sup_loss_true_weighted = (ce_loss(logits_x_lb, y_true, reduction='none') *w).sum().item()
                sup_loss_raw = ce_loss(logits_x_lb, y_lb, reduction='mean').mean().item()
            else:
                
                sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
                sup_loss_true = ce_loss(logits_x_lb, y_true, reduction='mean')
                sup_loss_true_weighted = 0.0
                sup_loss_raw = sup_loss.item()
                
            total_loss = sup_loss + self.lambda_u * unsup_loss
            lid_score = lid(feat_u,k=10,distmetric = 'l2')
            
        y_pred_lb = torch.max(logits_x_lb, dim=-1)[1].cpu().tolist()
        acc_lb = accuracy_score(y_lb.cpu().tolist(), y_pred_lb)        
        acc_lb_true = accuracy_score(y_true.cpu().tolist(), y_pred_lb)
        
        self.optimizer.zero_grad()
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

        for key, value in weight_log.items():
            try:
                self.log[key].append(value)
            except:
                print(key)
                
        return tb_dict,tb_meta, self.log
    


    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]
