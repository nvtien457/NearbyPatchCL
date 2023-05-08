from tqdm import tqdm
import os
from datetime import datetime
import shutil

import torch
import torchvision

from models import get_model
from datasets import get_dataset
from augmentations import get_aug
from losses import get_criterion
from optimizers import get_optimizer, get_scheduler
from tools import get_args, Logger, knn_monitor, AverageMeter
from trainer import Trainer

def main(args):
    train_loader = torch.utils.data.DataLoader(
        dataset=get_dataset(
            dataset_cfg=args.dataset,
            transform=get_aug(args.aug),
            **args.dataset_kwargs
        ),
        batch_size=args.train.batch_size,
        shuffle=True,
        **args.dataloader_kwargs
    )
    
    memory_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.ImageFolder(root=args.mem_dir, transform=get_aug(args.aug, train=False)),
        shuffle=False,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.ImageFolder(root=args.val_dir, transform=get_aug(args.aug, train=False)),
        shuffle=False,
        batch_size=args.train.batch_size,
        **args.dataloader_kwargs
    )

    print(train_loader.dataset.classes)
    print(val_loader.dataset.class_to_idx)

    model = get_model(model_cfg=args.model)
    scaler = torch.cuda.amp.GradScaler()
    model = model.to(args.device)           # MUST move to cuda before load checkpoint

    criterion = get_criterion(criterion_cfg=args.train.criterion)

    # Use Mixed Precision Training
    args.train.optimizer.params['lr']        = args.train.optimizer.params['lr']        * args.train.batch_size * args.train.iters_to_accumulate / 256
    args.train.scheduler.params['base_lr']   = args.train.scheduler.params['base_lr']   * args.train.batch_size * args.train.iters_to_accumulate / 256
    args.train.scheduler.params['warmup_lr'] = args.train.scheduler.params['warmup_lr'] * args.train.batch_size * args.train.iters_to_accumulate / 256
    args.train.scheduler.params['final_lr']  = args.train.scheduler.params['final_lr']  * args.train.batch_size * args.train.iters_to_accumulate / 256

    # Number of iteration per epoch
    iter_per_epoch = len(train_loader) // args.train.iters_to_accumulate
    args.train.scheduler.params['iter_per_epoch'] = iter_per_epoch

    optimizer = get_optimizer(optimizer_cfg=args.train.optimizer, model=model)
    scheduler = get_scheduler(scheduler_cfg=args.train.scheduler, optimizer=optimizer)
    print(len(scheduler.lr_schedule))

    # create log & ckpt
    args.log_dir = os.path.join(args.log_dir, 'in-progress_' + args.name + '_' + datetime.now().strftime('%m%d%H%M%S'))
    args.ckpt_dir = os.path.join(args.ckpt_dir, args.name)
    if os.path.exists(args.log_dir):
      print(f'use folder {args.log_dir}')
    else:
      os.makedirs(args.log_dir, exist_ok=False)
      print(f'creating folder {args.log_dir}')
    if os.path.exists(args.ckpt_dir):
      print(f'use folder {args.ckpt_dir}')
    else:
      os.makedirs(args.ckpt_dir, exist_ok=True)
      print(f'creating folder {args.ckpt_dir}')
    shutil.copy2(args.config_file, args.log_dir)

    start_epoch = 0
    best_loss = 99999999
    logger = Logger(log_dir=args.log_dir, tensorboard=args.tensorboard, matplotlib=args.matplotlib)

    if args.resume.status:
        print("=> loading history at '{}'".format(args.resume.ckpt))

        checkpoint = torch.load(args.resume.ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        scheduler.iter = start_epoch * iter_per_epoch
        # best_loss = checkpoint['best_loss']

        if args.resume.event is not None:
            logger.load_event(args.resume.event, checkpoint['epoch'])

        print("=> loaded checkpoint '{}' (epoch = {}, iter = {}, loss = {})".format(args.resume.ckpt, 
                                                                                checkpoint['epoch'], scheduler.iter, best_loss))

    # Start training
    global_progress = tqdm(range(start_epoch, args.train.stop_epoch), initial=start_epoch, total=args.train.stop_epoch-1, desc='Training')

    metric_meter = {
        'train_loss': AverageMeter('train_loss', ':.4e'),
        'train_acc': AverageMeter('train_acc', ':.2f'),
        'val_loss': AverageMeter('val_loss', ':.4e'),
        'val_acc': AverageMeter('val_acc', ':.2f')
    }
    for m in args.train.metrics:
        metric_meter[m] = AverageMeter(m, ':.2f')

    for epoch in global_progress:

        # Training
        model.train()

        for k, meter in metric_meter.items():
            metric_meter[k].reset()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            batch_size = targets.shape[0]
            
            # Runs the forward pass with autocasting.
            with torch.autocast(device_type=args.device, dtype=torch.float16):
                batch_size = targets.shape[0]

                inputs = inputs.to(args.device)
                targets = targets.to(args.device)

                # compute output
                preds = model(inputs)
                loss = criterion(preds, targets)

                data_dict = {
                    'train_loss': loss
                }

                labels = torch.argmax(torch.softmax(preds, dim=1), dim=1)
                data_dict['train_acc'] = torch.sum(torch.eq(labels, targets)) / batch_size * 100.00

                for m in args.train.metrics:
                    pass

                loss = data_dict['train_loss'] / args.train.iters_to_accumulate
                # print(loss)
                # break

            scaler.scale(loss).backward()

            if (batch_idx + 1) % args.train.iters_to_accumulate == 0:
                scaler.step(optimizer)
                scaler.update()

                if scheduler is not None:
                    scheduler.step()

                optimizer.zero_grad()

            # update metric meters
            for key, value in data_dict.items():
                # convert Tensor to value
                if isinstance(value, torch.Tensor):
                    data_dict[key] = value.item()

                if key in metric_meter.keys():
                    # print(key, value, batch_size)
                    metric_meter[key].update(value, n=batch_size)

            # update visual training infor
            data_dict['lr'] = optimizer.param_groups[0]['lr']
            logger.update_scalers(data_dict)


        # Evaluation
        model.eval()

        for batch_idx, (inputs, targets) in enumerate(val_loader):
            batch_size = targets.shape[0]

            with torch.no_grad():
                inputs = inputs.to(args.device)
                targets = targets.to(args.device)

                # compute output
                preds = model(inputs)
                loss = criterion(preds, targets)

                labels = torch.argmax(torch.softmax(preds, dim=1), dim=1)
                acc = torch.sum(torch.eq(labels, targets)) / batch_size * 100.00

                data_dict = {
                    'val_loss': loss,
                    'val_acc': acc
                }

                for m in args.train.metrics:
                    pass

            
            # update metric meters
            for key, value in data_dict.items():
                # convert Tensor to value
                if isinstance(value, torch.Tensor):
                    data_dict[key] = value.item()

                if key in metric_meter.keys():
                    # print(key, value, batch_size)
                    metric_meter[key].update(value, n=batch_size)
            
        epoch_dict = dict()
        for key, meter in metric_meter.items():
            epoch_dict[key] = meter.avg
        # break
        
        if args.train.knn_monitor and (epoch + 1) % args.train.knn_interval == 0:
            accuracy = knn_monitor(model.backbone, memory_loader, val_loader, args.device,
                            k=min(args.train.knn_k, len(memory_loader.dataset)),
                            hide_progress=args.hide_progress)
            
            epoch_dict['knn_acc'] = accuracy

        epoch_dict['epoch'] = epoch

        # Display
        global_progress.set_postfix(epoch_dict)
        logger.update_scalers(epoch_dict)

        # Save the checkpoint
        filepath = os.path.join(args.ckpt_dir, 'ckpt_{:03d}.pth'.format(epoch))
        checkpoint = {
            'backbone': args.model.backbone,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'best_loss': best_loss
        }

        if loss < best_loss:
            best_loss = loss
            checkpoint['best_loss'] = best_loss
            filepath = filepath.replace('ckpt_', 'ckpt_best_')
            torch.save(checkpoint, filepath)

        elif (epoch + 1) % args.train.save_interval == 0:
            torch.save(checkpoint, filepath)


if __name__=='__main__':
    args = get_args()
    print('Device:', args.device)

    main(args)

    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')

    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')
