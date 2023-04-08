import math

import torch
import torch.optim as optim

from .lars import LARS
from .lars_simclr import LARS_simclr
from .larc import LARC
from .lr_scheduler import LR_Scheduler

def get_optimizer(optimizer_cfg, model):
    name = optimizer_cfg.name
    lr = optimizer_cfg.params['lr']

    predictor_prefix = ('module.predictor', 'predictor')
    parameters = [{
        'name': 'base',
        'params': [param for name, param in model.named_parameters() if not name.startswith(predictor_prefix)],
        'lr': lr
    },{
        'name': 'predictor',
        'params': [param for name, param in model.named_parameters() if name.startswith(predictor_prefix)],
        'lr': lr
    }]

    if name == 'Adam':
        optimizer = optim.Adam(parameters, **optimizer_cfg.params)

    elif name == 'lars':
        optimizer = LARS(parameters, **optimizer_cfg.params)

    elif name == 'sgd':
        optimizer = torch.optim.SGD(parameters, **optimizer_cfg.params)

    elif name == 'lars_simclr': # Careful
        optimizer = LARS_simclr(model.named_modules(), **optimizer_cfg.params)

    elif name == 'larc':
        optimizer = LARC(
            torch.optim.SGD(
                parameters,
                **optimizer_cfg.params
            ),
            **optimizer_cfg.params
        )

    else:
        raise NotImplementedError

    return optimizer

def get_scheduler(scheduler_cfg, optimizer):
    if scheduler_cfg.name == 'lr_scheduler':
        return LR_Scheduler(optimizer=optimizer, **scheduler_cfg.params)

    else:
        return None


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.train.base_lr
    if args.train.optimizer.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.train.num_epochs))
    else:  # stepwise lr schedule
        for milestone in args.train.optimizer.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr