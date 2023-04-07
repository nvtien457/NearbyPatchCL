#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import builtins
import math
import os
import random
import shutil
import time
from datetime import datetime
import warnings
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from models import get_model
from augmentations import get_aug
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler, adjust_learning_rate
from tools import accuracy, get_args, Logger, AverageMeter, NT_Xent

def main(args):

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu

        dist.init_process_group(backend=args.dist_backend, store=dist.FileStore("./tmp/filestore", args.world_size),
                                world_size=args.world_size, rank=args.rank)

    ##########################################################################################

    
    train_dataset = get_dataset(dataset=args.dataset.name, 
        data_dir=args.data_dir,
        transform=get_aug(train=True, **args.aug_kwargs),
        train=True,
        **args.dataset_kwargs
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=args.train.batch_size, 
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **args.dataloader_kwargs
    )
    
    ########## CREATE MODEL ##########

    model = get_model(args.model, args.model.castrate)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    ##################################################################################################

    # define loss function (criterion) and optimizer
    criterion = NT_Xent(batch_size=args.train.batch_size, temperature=args.train.temperature, world_size=torch.cuda.device_count() * args.train.nodes)

    optimizer = get_optimizer(
        args.train.optimizer.name, model,
        # lr=args.train.base_lr * args.train.batch_size / 256,
        lr = 0.3 * args.train.batch_size / 256,
        momentum=args.train.optimizer.momentum,
        weight_decay=args.train.optimizer.weight_decay
    )

    # lr_scheduler = LR_Scheduler(
    #     optimizer,
    #     args.train.warmup_epochs, args.train.warmup_lr * args.train.batch_size / 256,
    #     args.train.num_epochs, args.train.base_lr * args.train.batch_size / 256,
    #                               args.train.final_lr * args.train.batch_size / 256,
    #     len(train_loader),
    #     constant_predictor_lr=True  # see the end of section 4.2 predictor
    # )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.train.num_epochs, eta_min=0, last_epoch=-1
        )

    ##################################################################################################

    name = args.name + datetime.now().strftime('_%d%m_%H%M%S')

    args.log_dir = os.path.join(args.log_dir, 'in-progress_' + name)

    os.makedirs(args.log_dir, exist_ok=False)
    print(f'creating file {args.log_dir}')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    shutil.copy2(args.config_file, args.log_dir)

    ckpt_folder = os.path.join(args.ckpt_dir, name)
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)
    print('Checkpoint is saved in', ckpt_folder)

    ##################################################################################################

    logger = Logger(tensorboard=args.logger.tensorboard, matplotlib=args.logger.matplotlib, log_dir=args.log_dir)
    best_loss = 99999999

    # optionally resume from a checkpoint
    if args.resume.status:
        print("=> loading history at '{}'".format(args.resume))

        if args.gpu is None:
            checkpoint = torch.load(args.resume.ckpt)
        else:
            # Map model to be loaded to specified single gpu.
            loc = args.gpu
            checkpoint = torch.load(args.resume.ckpt, map_location=loc)

        args.start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        # Because optimizer is created, scheduler must be created too
        # lr_scheduler = LR_Scheduler(
        #     optimizer,
        #     args.train.warmup_epochs, args.train.warmup_lr * args.train.batch_size / 256,
        #     args.train.num_epochs, args.train.base_lr * args.train.batch_size / 256,
        #                             args.train.final_lr * args.train.batch_size / 256,
        #     len(train_loader),
        #     constant_predictor_lr=True  # see the end of section 4.2 predictor
        # )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.num_epochs, eta_min=0, last_epoch=-1
        )

        lr_scheduler.iter = args.start_epoch * len(train_loader)
        lr_scheduler.current_lr = lr_scheduler.lr_schedule[lr_scheduler.iter]

        bess_loss = logger.load_event(checkpoint['epoch'])

        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    ##################################################################################################

    cudnn.benchmark = True

    ##################################################################################################

    progress = tqdm(range(args.start_epoch, args.train.num_epochs), desc=f'Training')
    for epoch in progress:
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        lr = optimizer.param_groups[0]["lr"]
        loss, acc1, acc5 = train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, logger, args)

        progress.set_postfix({'loss': ':.4f'.format(loss), 'acc@1': ':6.2f'.format(acc1), 'acc@5': ':6.2f'.format(acc5)})
        logger.update_scalers({'epoch': epoch, 'acc@1': acc1, 'acc@5': acc5})

        filename = os.path.join(ckpt_folder, 'ckpt_{:03d}.pth'.format(epoch))
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }

        if loss < best_loss:
            best_loss = loss
            filename.replace('ckpt_', 'ckpt_best_')
            torch.save(checkpoint, filename)

        elif (epoch + 1) % args.save_interval == 0:
            torch.save(checkpoint, filename)

    if args.distributed:
        dist.destroy_process_group()


def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, logger, args):    
    loss_meter = AverageMeter('Loss', ':.4e')
    top1_meter = AverageMeter('Acc@1', ':6.2f')
    top5_meter = AverageMeter('Acc@5', ':6.2f')

    # switch to train mode
    model.train()

    progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.train.num_epochs}')
    # iter_loss = []
    for i, ((x_i, x_j), _) in enumerate(progress):
        batch_size = x_i.shape[0]

        if args.gpu is not None:
            x_i = x_j.cuda(args.gpu, non_blocking=True)
            x_j = x_j.cuda(args.gpu, non_blocking=True)

        optimizer.zero_grad()

        # compute output
        h_i, z_i = model(x_i)
        h_j, z_j = model(x_j)
        loss = criterion(z_i, z_j)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(torch.mm(z_i, z_j.T), torch.arange(0, batch_size, dtype=torch.long).cuda(), topk=(1, 5))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()

        # iter_loss.append(loss.item())
        loss_meter.update(loss.item(), batch_size)
        top1_meter.update(acc1.item(), batch_size)
        top5_meter.update(acc5.item(), batch_size)

        data_dict = {
            'lr': optimizer.param_groups[0]['lr'],
            'loss': loss_meter.avg,
            'acc@1': top1_meter.avg,
            'acc@5': top5_meter.avg,
        }

        progress.set_postfix(data_dict)
        logger.update_scalers({'lr': optimizer.param_groups[0]['lr'], 'loss': loss_meter.avg,})

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


if __name__ == '__main__':
    args = get_args()

    main(args)

    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')

    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')