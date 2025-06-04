# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import copy 
import torch
import math
from semilearn.core import AlgorithmBase
from semilearn.algorithms.hooks import PseudoLabelingHook, FixedThresholdingHook, MetaGradientHook
from semilearn.algorithms.utils import ce_loss, consistency_loss, SSL_Argument, gradient_match, lid, TwoNN,manifold_loss_soft,manifold_loss_soft_exp,mixup_one_target
from sklearn.metrics import confusion_matrix,roc_auc_score,f1_score,accuracy_score, balanced_accuracy_score
import torch.nn.functional as F

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
        self.init(T=args.T, p_cutoff=args.p_cutoff, log=args.log, tsa_schedule=args.tsa_schedule, mixup_alpha=args.mixup_alpha)

    def init(self, T, p_cutoff, log,tsa_schedule='none',mixup_alpha=0):
        self.T = T
        self.p_cutoff = p_cutoff
        self.tsa_schedule = tsa_schedule
        self.log=log
        self.mixup_alpha = mixup_alpha
        
    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")

        super().set_hooks()
    def train_step(self, idx_lb,x_lb, y_lb,y_true, idx_ulb, x_ulb_w, x_ulb_s, y_ulb_true):
        num_lb = y_lb.shape[0]
        # inference and calculate sup/unsup losses
        with self.amp_cm():
            
            if self.mixup_alpha>0:
                if y_lb.shape != self.num_classes:
                    
                    y_lb = F.one_hot(y_lb, self.num_classes)

                x_lb,y_lb , _= mixup_one_target(x_lb, y_lb, alpha=self.mixup_alpha, is_bias=True)
          
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
            
            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_w)
            outs_x_ulb_w = self.model(x_ulb_w)
            logits_x_ulb_w = outs_x_ulb_w['logits']
            probs_x_ulb = torch.softmax(logits_x_ulb_w, dim=-1)
            max_probs, y = torch.max(probs_x_ulb, dim=-1)
            flag = torch.sum(max_probs.ge(0.7)).item()>0                    
            # if self.args.meta_updating and self.it>self.args.s_it:
            

            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
            sup_loss_true = ce_loss(logits_x_lb, y_true, reduction='mean')



            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=logits_x_ulb_w,
                                          use_hard_label=False,
                                          T=self.T)
            
            
            unsup_loss = consistency_loss(logits_x_ulb_s,
                                          pseudo_label,
                                          'ce',
                                          mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss
            # total_loss = sup_loss 

        lid_score = lid(feature_u_w,k=10,distmetric = 'l2')  
        # parameter updates
        self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)
        

        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/sup_loss_true'] = sup_loss_true.item()
        tb_dict['train/unsup_loss'] = unsup_loss.item()
        tb_dict['train/total_loss'] = total_loss.item()
        tb_dict['train/mask_ratio'] = mask.float().mean().item()
        tb_dict['train/feat_u'] = lid_score.item()

        
        return tb_dict

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
