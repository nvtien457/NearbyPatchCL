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
    parser.add_argument('--data_dir', type=str, default='../data')
    # parser.add_argument('--log_dir', type=str, default='../logs')
    parser.add_argument('--ckpt_dir', type=str, default='../checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval_from', type=str, default=None)
    parser.add_argument('--hide_progress', action='store_true')

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
            if value == 'None':
                vars(args)[key] = None
            else:
                vars(args)[key] = value

    vars(args)['start_epoch'] = 0

    if args.debug:
        if args.train: 
            args.train.batch_size
            args.train.num_epochs = 1
            args.train.stop_at_epoch = 1
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

    args.log_dir = os.path.join(args.log_dir, 'in-progress_' + datetime.now().strftime('%m%d%H%M%S_') + args.name)

    os.makedirs(args.log_dir, exist_ok=False)
    print(f'creating file {args.log_dir}')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    shutil.copy2(args.config_file, args.log_dir)

    set_deterministic(args.seed)


    vars(args)['aug_kwargs'] = {
        # 'name':args.model.name,
        'name': args.aug.name,
        'version': args.aug.version,
        'image_size': args.dataset.image_size
    }
    vars(args)['dataset_kwargs'] = {
        'download':args.download,
        'debug_subset_size': args.debug_subset_size if args.debug else None,
    }
    vars(args)['dataloader_kwargs'] = {
        'drop_last': True,
        'pin_memory': True,
        'num_workers': args.dataset.num_workers,
    }

    return args
