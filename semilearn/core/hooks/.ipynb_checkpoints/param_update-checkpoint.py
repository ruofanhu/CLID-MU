# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from .hook import Hook


class ParamUpdateHook(Hook):
    def __init__(self) -> None:
        super().__init__()
    
    # specific param_update function, called inside train_step of each algorithm
    def param_update(self, algorithm, loss):
        # algorithm.optimizer.zero_grad()
        # update parameters
        if algorithm.use_amp:
            algorithm.loss_scaler.scale(loss).backward()
            if (algorithm.clip_grad > 0):
                algorithm.loss_scaler.unscale_(algorithm.optimizer)
                torch.nn.utils.clip_grad_norm_(algorithm.model.parameters(), algorithm.clip_grad)
            algorithm.loss_scaler.step(algorithm.optimizer)
            algorithm.loss_scaler.update()
        else:
            loss.backward()
            if (algorithm.clip_grad > 0):
                torch.nn.utils.clip_grad_norm_(algorithm.model.parameters(), algorithm.clip_grad)
            
            ####
            for name, param in algorithm.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_magnitude = param.grad.norm().item()
                    algorithm.gradient_magnitudes[name].append(grad_magnitude)
            

            torch.save(algorithm.gradient_magnitudes,f'./mg_gradient_{algorithm.args.net}.pt')
            #############

            algorithm.optimizer.step()
        if algorithm.scheduler is not None:
            algorithm.scheduler.step()

        algorithm.model.zero_grad()