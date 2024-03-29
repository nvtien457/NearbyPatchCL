from tqdm import tqdm

import torch

from .simclr import simclr_train
from .moco import moco_train
from .simsiam import simsiam_train
from .byol import byol_train
from .simtriplet import simtriplet_train
from .supcon import supcon_train
from .clsa import clsa_train
from .barlow_twins import barlowtwins_train
from .barlow_twins_nearby import barlowtwins_nearby_train
from .micle import micle_train
from .swav import swav_train

import sys
sys.path.append('../')
from tools import AverageMeter, knn_monitor

class Trainer:
    def __init__(self, train_loader, model, scaler, criterion, optimizer, scheduler, logger, args):
        self.train_loader = train_loader
        self.model = model
        self.scaler = scaler
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.args = args

        self.metric_meter = {
            'loss': AverageMeter('loss', ':.4e'),
        }
        for m in args.train.metrics:
            self.metric_meter[m] = AverageMeter(m, ':.2f')


        if args.model.name == 'simclr':
            self.train_func = simclr_train
        
        elif args.model.name == 'moco':
            self.train_func = moco_train

        elif args.model.name == 'simsiam':
            self.train_func = simsiam_train

        elif args.model.name == 'simtriplet':
            self.train_func = simtriplet_train

        elif args.model.name == 'byol':
            self.train_func = byol_train

        elif args.model.name == 'supcon':
            self.train_func = supcon_train

        elif args.model.name == 'swav':
            self.train_func = swav_train 

        elif args.model.name == 'clsa':
            self.train_func = clsa_train
        elif args.model.name == 'micle':
            self.train_func = micle_train

        elif args.model.name == 'barlow_twins':
            self.train_func = barlowtwins_train
        elif args.model.name == 'barlow_twins_nearby':
            self.train_func = barlowtwins_nearby_train
        else:
            raise NotImplementedError

    def train(self, epoch):
        # swith to train mode
        self.model.train()

        for k, meter in self.metric_meter.items():
            self.metric_meter[k].reset()

        progress = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.args.train.num_epochs}', disable=self.args.hide_progress)
        for batch_idx, (inputs, targets) in enumerate(progress):
            batch_size = targets.shape[0]
            
            # Runs the forward pass with autocasting.
            with torch.autocast(device_type=self.args.device, dtype=torch.float16):
                data_dict = self.train_func(inputs, targets, self.model, self.criterion, self.args)
                loss = data_dict['loss'] / self.args.train.iters_to_accumulate
                # print(loss)
                # break

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.args.train.iters_to_accumulate == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.optimizer.zero_grad()

            # update metric meters
            for key, value in data_dict.items():
                # convert Tensor to value
                if isinstance(value, torch.Tensor):
                    data_dict[key] = value.item()

                if key in self.metric_meter.keys():
                    # print(key, value, batch_size)
                    self.metric_meter[key].update(value, n=batch_size)

            # update visual training infor
            data_dict['lr'] = self.optimizer.param_groups[0]['lr']

            progress.set_postfix(data_dict)
            self.logger.update_scalers(data_dict)

        epoch_dict = dict()
        for key, meter in self.metric_meter.items():
            epoch_dict[key + '_avg'] = meter.avg

        return epoch_dict