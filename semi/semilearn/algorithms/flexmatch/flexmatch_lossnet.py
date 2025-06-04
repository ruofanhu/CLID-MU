
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from .utils import FlexMatchThresholdingHook

from semilearn.core import AlgorithmBase
from semilearn.algorithms.hooks import PseudoLabelingHook,MetaLossNetHook
from semilearn.algorithms.utils import ce_loss, consistency_loss, SSL_Argument, str2bool
from sklearn.metrics import accuracy_score



class FlexMatch_lossnet(AlgorithmBase):
    """
        FlexMatch algorithm (https://arxiv.org/abs/2110.08263).

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
            - ulb_dest_len (`int`):
                Length of unlabeled data
            - thresh_warmup (`bool`, *optional*, default to `True`):
                If True, warmup the confidence threshold, so that at the beginning of the training, all estimated
                learning effects gradually rise from 0 until the number of unused unlabeled data is no longer
                predominant

        """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # flexmatch specificed arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, log=args.log ,hard_label=args.hard_label, thresh_warmup=args.thresh_warmup)
    
    def init(self, T, p_cutoff, log,hard_label=True, thresh_warmup=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.thresh_warmup = thresh_warmup
        self.log=log
        self.gradients = {}
    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FlexMatchThresholdingHook(ulb_dest_len=self.args.ulb_dest_len, num_classes=self.num_classes, thresh_warmup=self.args.thresh_warmup), "MaskingHook")
        self.register_hook(MetaLossNetHook(), "MetaLossNetHook")

        super().set_hooks()

    def train_step(self, idx_lb,x_lb, y_lb,y_true, idx_ulb, x_ulb_w, x_ulb_s, y_ulb_true):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            meta_hook = MetaLossNetHook()
            tb_meta, weight_log, acc_val = meta_hook.cal_meta_gradient(idx=idx_lb,lb_data=x_lb, lb_label=y_lb,\
                                                              y_true=y_true, algorithm=self)
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

            # sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_w, idx_ulb=idx_ulb)
            
            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=logits_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T)

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

        # parameter updates
        y_pred_lb = torch.max(logits_x_lb, dim=-1)[1].cpu().tolist()
        acc_lb = accuracy_score(y_lb.cpu().tolist(), y_pred_lb)        
        acc_lb_true = accuracy_score(y_true.cpu().tolist(), y_pred_lb)
        
        sup_loss_all = ce_loss(logits_x_lb, y_lb, reduction='none')
        mislabeled_ = (y_true!=y_lb).long().cpu().tolist() # True means mislabeled sample
        self.epoch_loss[self.epoch]['loss'].extend(sup_loss_all.cpu().tolist())
        self.epoch_loss[self.epoch]['flag'].extend(mislabeled_)     
        
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
        tb_dict['train/mask_ratio'] = mask.float().mean().item()
        for key, value in weight_log.items():
            try:
                self.log[key].append(value)
            except:
                print(key)
                
        return tb_dict,tb_meta, self.log
    

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['classwise_acc'] = self.hooks_dict['MaskingHook'].classwise_acc.cpu()
        save_dict['selected_label'] = self.hooks_dict['MaskingHook'].selected_label.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['MaskingHook'].classwise_acc = checkpoint['classwise_acc'].cuda(self.gpu)
        self.hooks_dict['MaskingHook'].selected_label = checkpoint['selected_label'].cuda(self.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--thresh_warmup', str2bool, True),
        ]
