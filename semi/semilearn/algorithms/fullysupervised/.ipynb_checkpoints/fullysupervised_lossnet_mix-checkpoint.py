# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F

from semilearn.core import AlgorithmBase
from semilearn.algorithms.utils import ce_loss, mixup_one_target
from semilearn.algorithms.hooks import MetaLossNetHook
import os
from sklearn.metrics import confusion_matrix,roc_auc_score,f1_score,accuracy_score, balanced_accuracy_score

class FullySupervised_lossnet_mix(AlgorithmBase):
    """
        Train a fully supervised model using labeled data only. This serves as a baseline for comparison.

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
        """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)

    def set_hooks(self):
        self.register_hook(MetaLossNetHook(), "MetaLossNetHook")
        super().set_hooks()
    
    
    
    # def train_step(self, x_lb, y_lb):
    def train_step(self, idx_lb,x_lb, y_lb,y_true):

        # inference and calculate sup/unsup losses
        
        with self.amp_cm():

            if self.args.mixup_alpha:
                x_lb, y_lb, _ = mixup_one_target(inputs, input_labels,
                                                 self.args.mixup_alpha,
                                                 is_bias=False)
                
            logits_x_lb = self.model(x_lb)['logits']
            
            if self.args.meta_goal is not None:
                meta_hook = MetaLossNetHook()
                tb_meta, weight_log, acc_val = meta_hook.cal_meta_gradient(idx=idx_lb,lb_data=x_lb, lb_label=y_lb,\
                                                                  y_true=y_true, algorithm=self)    
                
                # sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
                cost_w= ce_loss(logits_x_lb, y_lb, reduction='none')
                cost_v=torch.reshape(cost_w, (len(cost_w), 1))
                with torch.no_grad():
                    w = self.wnet(cost_v.data)
                
                # norm_c = torch.sum(w)
                # if norm_c != 0:
                #     w_norm = w / norm_c
                # print(w_norm)
                weight_log['w_lid_feat_l2']= torch.squeeze(w).detach().cpu()
            
            # print(w_norm)
            # self.args.meta_goal = None if self.args.meta_goal == 'None' else self.args.meta_goal
            # if self.args.meta_goal is not None:

                sup_loss = torch.mean(ce_loss(logits_x_lb, y_lb, reduction='none')*torch.squeeze(w))
                # sup_loss = torch.sum(cost_v*w_norm)
                sup_loss_true = ce_loss(logits_x_lb, y_true, reduction='mean')
                sup_loss_true_weighted = (ce_loss(logits_x_lb, y_true, reduction='none') *w).sum().item()
                sup_loss_raw = ce_loss(logits_x_lb, y_lb, reduction='mean').item()
                for key, value in weight_log.items():
                    self.log[key].append(value)
            else:
                
                sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
                sup_loss_true = ce_loss(logits_x_lb, y_true, reduction='mean')
                sup_loss_true_weighted = 0.0
                sup_loss_raw = sup_loss.item()
                tb_meta={'lid_80':{}}
                acc_val=0
                
        
        y_pred_lb = torch.max(logits_x_lb, dim=-1)[1].cpu().tolist()
        # for id,weight in zip(idx_lb, torch.squeeze(w)):
        #     y_id_weight[id]=weight
        #     if id in self.weight_dict:
        #         self.weight_dict[id]=self.args.beta*self.weight_dict[id]+(1-self.args.beta)*weight
        #     else:
        #         self.weight_dict[id]=weight
                
        if self.args.beta>0:
            print(self.args.beta)
            if logits_x_lb.shape != y_lb.shape:
                input_labels = F.one_hot(y_lb, self.num_classes)
            else:
                input_labels = y_lb.copy()
            
            
            y_pred_lb_s = w*input_labels+(1-w)*F.softmax(logits_x_lb, dim=1)
            sup_loss_ = self.args.beta * ce_loss(y_pred_lb_s, y_lb, reduction='mean')
            print(ce_loss(y_pred_lb_s, y_lb, reduction='mean').item())
            y_pred_lb = torch.max(y_pred_lb_s, dim=-1)[1].cpu().tolist()
            
        acc_lb = accuracy_score(y_lb.cpu().tolist(), y_pred_lb)        
        acc_lb_true = accuracy_score(y_true.cpu().tolist(), y_pred_lb) 
        
        sup_loss_all_raw = ce_loss(logits_x_lb, y_lb, reduction='none')
        mislabeled_ = (y_true!=y_lb).long().cpu().tolist() # True means mislabeled sample
        self.epoch_loss[self.epoch]['loss'].extend(sup_loss_all_raw.cpu().tolist())
        self.epoch_loss[self.epoch]['flag'].extend(mislabeled_)
        
        # parameter updates
        self.optimizer.zero_grad()
        self.call_hook("param_update", "ParamUpdateHook", loss=sup_loss)
        


        # tensorboard_dict update
        tb_dict = {}
        tb_dict['train/acc_lb'] = acc_lb
        tb_dict['train/acc_lb_true'] = acc_lb_true
        tb_dict['train/sup_loss'] = sup_loss_raw
        tb_dict['train/sup_loss_weighted'] = sup_loss.item()
        tb_dict['train/sup_loss_true'] = sup_loss_true.item()
        tb_dict['train/sup_loss_true_weighted'] = sup_loss_true_weighted
        tb_dict['train_val_acc'] = acc_val

        return tb_dict, tb_meta, self.log

    
    def train(self):
        # lb: labeled, ulb: unlabeled
        save_path = os.path.join(self.args.save_dir, self.args.save_name)
        log_path = os.path.join(save_path, 'data_weight_log.pt')
        logloss_path = os.path.join(save_path, 'loss_log.pt')

        self.model.train()
        self.wnet.train()
        self.call_hook("before_run")
            
        for epoch in range(self.epochs):
            self.epoch = epoch
            self.epoch_loss[self.epoch]={'loss':[],'flag':[]}
            # prevent the training iterations exceed args.num_train_iter
            if self.it > self.num_train_iter:
                break
                
            # lr = self.args.lr * ((0.1 ** int(self.it >= 18000)) * (0.1 ** int(self.it >= 19000)))  # For WRN-28-10

            # lr = self.args.lr * (0.1 ** int(self.it >= 18000))   # For WRN-28-10
            # # if self.it >= 2000:
            # #     lr=self.args.lr/10
            # for param_group in self.optimizer.param_groups:
            #     param_group['lr'] = lr
                
            #lr = args.lr * ((0.1 ** int(iters >= 20000)) * (0.1 ** int(iters >= 25000)))  # For ResNet32
            # for param_group in self.optimizer.param_groups:
            #     param_group['lr'] = lr
                
            self.call_hook("before_train_epoch")

            for data_lb in self.loader_dict['train_lb']:
                try:
                    data_eval=next(data_eval_iter)
                except:
                    data_eval_iter = iter(self.loader_dict['meta'])
                    data_eval = next(data_eval_iter) 
                # prevent the training iterations exceed args.num_train_iter
                if self.it > self.num_train_iter:
                    break
                # self.tb_dict,self.tb_meta, self.log = self.train_step(**self.process_batch(data_eval,**data_lb, **data_ulb))

                self.call_hook("before_train_step")
                self.tb_dict,self.tb_meta, self.log = self.train_step(**self.process_batch(**data_lb))
                
                self.call_hook("after_train_step")
                self.it += 1
                
                if self.it%64 ==0:
                    torch.save(self.log, log_path)

            self.call_hook("after_train_epoch")
            torch.save(self.epoch_loss, logloss_path)
        self.call_hook("after_run")
        
