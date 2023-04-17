import argparse
import os
import torch

import numpy as np
import torch
import random

import re 
import yaml

import shutil
import warnings

from datetime import datetime


class Namespace(object):
    def __init__(self, somedict):
        for key, value in somedict.items():
            assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                if value == 'None':
                    self.__dict__[key] = None
                else:
                    self.__dict__[key] = value
    
    def __getattr__(self, attribute):

        raise AttributeError(f"Can not find {attribute} in namespace. Please write {attribute} in your config file(xxx.yaml)!")


def set_deterministic(seed):
    # seed by default is None 
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', required=True, type=str, help="xxx.yaml")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--debug_subset_size', type=int, default=8)
    parser.add_argument('--download', action='store_true', help="if can't find dataset, download from web")
    parser.add_argument('--data_dir', type=str, default='/content')
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--mem_dir', type=str, default='/content/TRAIN_VAL_SET')
    parser.add_argument('--val_dir', type=str, default='/content/VAL_SET')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval_from', type=str, default=None)
    parser.add_argument('--hide_progress', action='store_true')
    parser.add_argument('--mixed_precision', '-mp', action='store_true', help='Mixed precision traing')

    parser.add_argument('--dist-url', default='127.0.0.1', type=str,
                            help='url used to set up distributed training')
    parser.add_argument('--world-size', default=-1, type=int,
                            help='number of nodes for distributed training')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                            help='Use multi-processing distributed training to launch '
                                'N processes per node, which has N GPUs. This is the '
                                'fastest way to use PyTorch for either single node or '
                                'multi node data parallel training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
    parser.add_argument('--rank', default=-1, type=int,
                            help='node rank for distributed training')

    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
            vars(args)[key] = value

    vars(args)['start_epoch'] = 0

    if args.debug:
        if args.train: 
            args.train.batch_size = 2
            args.train.num_epochs = 5
            args.train.stop_epoch = 5
            args.train.scheduler.params.warmup_epochs = 2
        if args.eval: 
            args.eval.batch_size = 2
            args.eval.num_epochs = 1 # train only one epoch
        args.dataset.num_workers = 0

    if args.device.startswith('cuda'):
        cuda = args.device
        if cuda == 'cuda':
            cuda = cuda + ':0'

        available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
        if len(available_gpus) > 1:
            vars(args)['gpu'] = None
        else:
            vars(args)['gpu'] = cuda
    else:
        vars(args)['gpu'] = None

    assert not None in [args.log_dir, args.data_dir, args.ckpt_dir, args.name]

    # args.log_dir = os.path.join(args.log_dir, 'in-progress_' + datetime.now().strftime('%m%d%H%M%S_') + args.name)

    # os.makedirs(args.log_dir, exist_ok=False)
    # print(f'creating file {args.log_dir}')
    # os.makedirs(args.ckpt_dir, exist_ok=True)

    # shutil.copy2(args.config_file, args.log_dir)

    set_deterministic(args.seed)

    # model.params
    if args.model.params == None:
        vars(args.model)['params'] = dict()
    model_params = dict()
    for k, v in args.model.params.__dict__.items():
        model_params[k] = v
    vars(args.model)['params'] = model_params


    # criterion.params
    if args.train.criterion.params == None:
        vars(args.train.criterion)['params'] = dict()
    else:
        criterion_params = dict()
        for k, v in args.train.criterion.params.__dict__.items():
            criterion_params[k] = v
        vars(args.train.criterion)['params'] = criterion_params


    # optimizer.params
    if args.train.optimizer.params == None:
        vars(args.train.optimizer)['params'] = dict()
    else:
        optimizer_params = dict()
        for k, v in args.train.optimizer.params.__dict__.items():
            optimizer_params[k] = v
        vars(args.train.optimizer)['params'] = optimizer_params

    if args.eval.optimizer.params == None:
        vars(args.eval.optimizer)['params'] = dict()
    else:
        optimizer_params = dict()
        for k, v in args.eval.optimizer.params.__dict__.items():
            optimizer_params[k] = v
        vars(args.eval.optimizer)['params'] = optimizer_params


    # scheduler.params
    if args.train.scheduler.params == None:
        vars(args.train.scheduler)['params'] = dict()
    else:
        scheduler_params = dict()
        for k, v in args.train.scheduler.params.__dict__.items():
            scheduler_params[k] = v
        scheduler_params['base_lr'] = optimizer_params['lr']
        scheduler_params['num_epochs'] = args.train.num_epochs
        vars(args.train.scheduler)['params'] = scheduler_params

    if args.eval.scheduler.params == None:
        vars(args.eval.scheduler)['params'] = dict()
    else:
        scheduler_params = dict()
        for k, v in args.eval.scheduler.params.__dict__.items():
            scheduler_params[k] = v
        scheduler_params['base_lr'] = optimizer_params['lr']
        scheduler_params['num_epochs'] = args.eval.num_epochs
        vars(args.eval.scheduler)['params'] = scheduler_params

    # datset.params
    if args.dataset.params == None:
        vars(args.dataset)['params'] = {
            'data_dir': args.data_dir
        }
    else:
        dataset_params = dict()
        for k, v in args.dataset.params.__dict__.items():
            dataset_params[k] = v
        dataset_params['data_dir'] = args.data_dir
        dataset_params['name'] = str(dataset_params['name'])
        vars(args.dataset)['params'] = dataset_params

    # aug.params
    if args.aug.params == None:
        vars(args.aug)['params'] = dict()
    else:
        aug_params = dict()
        for k, v in args.aug.params.__dict__.items():
            aug_params[k] = v
        vars(args.aug)['params'] = aug_params

    vars(args)['dataset_kwargs'] = {
        'debug_subset_size': args.debug_subset_size if args.debug else None,
    }

    vars(args)['dataloader_kwargs'] = {
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.num_workers,
    }

    return args
