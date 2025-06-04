# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch
from semilearn.core import AlgorithmBase
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook,MetaLossNetHook
from semilearn.algorithms.utils import *
from sklearn.metrics import accuracy_score


class PseudoLabel_lossnet(AlgorithmBase):
    """
        Pseudo Label algorithm (https://arxiv.org/abs/1908.02983).

        Args:
        - args (`argparse`):
            algorithm arguments
        - net_builder (`callable`):
            network loading function
        - tb_log (`TBLog`):
            tensorboard logger
        - logger (`logging.Logger`):
            logger to use
        - p_cutoff(`float`):
            Confidence threshold for generating pseudo-labels
        - unsup_warm_up (`float`, *optional*, defaults to 0.4):
            Ramp up for weights for unsupervised loss
    """

    def __init__(self, args, net_builder, tb_log=None, logger=None, **kwargs):
        super().__init__(args, net_builder, tb_log, logger, **kwargs)
        self.init(p_cutoff=args.p_cutoff, unsup_warm_up=args.unsup_warm_up,log=args.log)

    def init(self, p_cutoff, unsup_warm_up=0.4,log=None):
        self.p_cutoff = p_cutoff
        self.unsup_warm_up = unsup_warm_up 
        self.log = log
        
    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        self.register_hook(MetaLossNetHook(), "MetaLossNetHook")

        super().set_hooks()

    def train_step(self, idx_lb,x_lb, y_lb,y_true, idx_ulb, x_ulb_w, x_ulb_s, y_ulb_true):
        # inference and calculate sup/unsup losses
        num_lb = y_lb.shape[0]

        with self.amp_cm():
            
            meta_hook = MetaLossNetHook()
            tb_meta, weight_log, acc_val = meta_hook.cal_meta_gradient(idx=idx_lb,lb_data=x_lb, lb_label=y_lb,\
                                                              y_true=y_true, algorithm=self)
            outs_x_lb = self.model(x_lb)
            logits_x_lb = outs_x_lb['logits']
            # calculate BN only for the first batch
            self.bn_controller.freeze_bn(self.model)
            outs_x_ulb = self.model(x_ulb_w)
            logits_x_ulb = outs_x_ulb['logits']
            self.bn_controller.unfreeze_bn(self.model)

            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=logits_x_ulb,
                                          use_hard_label=True)

            unsup_loss = consistency_loss(logits_x_ulb,
                                          pseudo_label,
                                          'ce',
                                          mask=mask)
            
            cost_w= ce_loss(logits_x_lb, y_lb, reduction='none')
            
            cost_v=torch.reshape(cost_w, (len(cost_w), 1))
            with torch.no_grad():
                # w = self.wnet(logits_x_lb.softmax(1))

                w = self.wnet(cost_v.data)
            
            weight_log['w_lid_feat_l2']= torch.squeeze(w).detach().cpu()
            
            self.args.meta_goal = None if self.args.meta_goal == 'None' else self.args.meta_goal
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
                
            unsup_warmup = np.clip(self.it / (self.unsup_warm_up * self.num_train_iter),  a_min=0.0, a_max=1.0)
            total_loss = sup_loss + self.lambda_u * unsup_loss * unsup_warmup

        # parameter updates

        y_pred_lb = torch.max(logits_x_lb, dim=-1)[1].cpu().tolist()
        acc_lb = accuracy_score(y_lb.cpu().tolist(), y_pred_lb)        
        acc_lb_true = accuracy_score(y_true.cpu().tolist(), y_pred_lb)   
        # parameter updates
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
            
            self.log[key].append(value)
        
        return tb_dict,tb_meta, self.log

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--unsup_warm_up', float, 0.4, 'warm up ratio for unsupervised loss'),
            # SSL_Argument('--use_flex', str2bool, False),
        ]