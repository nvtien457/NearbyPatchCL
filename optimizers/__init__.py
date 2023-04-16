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

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, optimizer, epoch, batch_id, total_batches):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr