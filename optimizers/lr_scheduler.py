import torch
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class LR_Scheduler(LRScheduler):
    def __init__(self, optimizer:Optimizer, 
                 num_epochs:int, iter_per_epoch:int, warmup_epochs:int,
                 warmup_lr:float, base_lr:float, final_lr:float, 
                 constant_predictor_lr=False):
        
        assert num_epochs > 0 and iter_per_epoch > 0 and warmup_epochs >= 0
        assert warmup_epochs <= num_epochs
        assert warmup_lr >= 0 and base_lr >= 0 and final_lr >= 0
      
        self.base_lr = base_lr
        self.constant_predictor_lr = constant_predictor_lr

        warmup_iter = iter_per_epoch * warmup_epochs
        warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
        decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
        cosine_lr_schedule = final_lr + 0.5*(base_lr - final_lr) * (1 + np.cos(np.pi*np.arange(decay_iter) / decay_iter))
      
        self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        self.optimizer = optimizer
        self.iter = -1
        self.current_lr = 0
        
        super().__init__(optimizer)

    def step(self):
        for param_group in self.optimizer.param_groups:

            if self.constant_predictor_lr and param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                lr = param_group['lr'] = self.lr_schedule[self.iter]
        
        self.iter += 1
        self.current_lr = lr
        return lr
        
    def get_lr(self):
        return self.current_lr