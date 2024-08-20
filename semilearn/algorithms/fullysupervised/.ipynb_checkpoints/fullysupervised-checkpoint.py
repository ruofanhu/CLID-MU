# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from semilearn.core import AlgorithmBase
from semilearn.algorithms.utils import ce_loss
import os

class FullySupervised(AlgorithmBase):
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

    def train_step(self, idx_lb,x_lb, y_lb,y_true):
        # inference and calculate sup/unsup losses
        with self.amp_cm():

            logits_x_lb = self.model(x_lb)['logits']

            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
            sup_loss_raw = ce_loss(logits_x_lb, y_lb, reduction='none')
            mislabeled_ = (y_true!=y_lb).long().cpu().tolist() # True means mislabeled sample
            self.epoch_loss[self.epoch]['loss'].extend(sup_loss_raw.cpu().tolist())
            self.epoch_loss[self.epoch]['flag'].extend(mislabeled_)

        # parameter updates
        self.call_hook("param_update", "ParamUpdateHook", loss=sup_loss)

        # tensorboard_dict update
        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        return tb_dict

    
    def train(self):
        # lb: labeled, ulb: unlabeled
        self.model.train()
        self.call_hook("before_run")
        save_path = os.path.join(self.args.save_dir, self.args.save_name)
        log_path = os.path.join(save_path, 'data_weight_log.pt')
        
        for epoch in range(self.epochs):
            self.epoch = epoch
            self.epoch_loss[epoch]={'loss':[],'flag':[]}
            # prevent the training iterations exceed args.num_train_iter
            if self.it > self.num_train_iter:
                break

            self.call_hook("before_train_epoch")

            for data_lb in self.loader_dict['train_lb']:

                # prevent the training iterations exceed args.num_train_iter
                if self.it > self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.tb_dict = self.train_step(**self.process_batch(**data_lb))
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")
                            
            torch.save(self.epoch_loss, log_path)
        self.call_hook("after_run")
