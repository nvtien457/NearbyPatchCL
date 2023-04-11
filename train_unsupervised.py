from tqdm import tqdm
import os
from datetime import datetime
import shutil

import torch
import torch.nn as nn
import torch.utils.data as data

from models import get_model
from datasets import get_dataset
from augmentations import get_aug
from losses import get_criterion
from optimizers import get_optimizer, get_scheduler
from tools import get_args, Logger
from trainer import Trainer

def main(args):
    train_loader = data.DataLoader(
        dataset=get_dataset(
            dataset_cfg=args.dataset,
            transform=get_aug(args.aug),
            **args.dataset_kwargs
        ),
        batch_size=args.train.batch_size,
        shuffle=True,
        **args.dataloader_kwargs
    )

    model = get_model(model_cfg=args.model)
    scaler = torch.cuda.amp.GradScaler()

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

    # create log & ckpt
    args.log_dir = os.path.join(args.log_dir, 'in-progress_' + args.name + '_' + datetime.now().strftime('%m%d%H%M%S'))
    args.ckpt_dir = os.path.join(args.ckpt_dir, args.name)
    os.makedirs(args.log_dir, exist_ok=False)
    print(f'creating folder {args.log_dir}')
    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f'creating folder {args.ckpt_dir}')
    shutil.copy2(args.config_file, args.log_dir)

    start_epoch = 0
    best_loss = 99999999
    logger = Logger(log_dir=args.log_dir, tensorboard=args.tensorboard, matplotlib=args.matplotlib)

    if args.resume.status:
        print("=> loading history at '{}'".format(args.resume.ckpt))

        checkpoint = torch.load(args.resume.ckpt)

        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        best_loss = logger.load_event(args.resume.event, checkpoint['epoch'])

        print("=> loaded checkpoint '{}' (epoch = {}, iter = {}, loss = {})".format(args.resume, checkpoint['epoch'], lr_scheduler.iter, best_loss))

    # Start training
    global_progress = tqdm(range(start_epoch, args.train.stop_epochs), initial=start_epoch, total=args.train.stop_epochs, desc='Training')

    model = model.to(args.device)
    trainer = Trainer(train_loader=train_loader, model=model, scaler=scaler,
                    criterion=criterion, optimizer=optimizer, scheduler=scheduler, 
                    logger=logger, args=args)

    for epoch in global_progress:

        # Training
        metrics = trainer.train(epoch)
        loss = metrics['loss_avg']

        # Display
        global_progress.set_postfix(metrics)
        logger.update_scalers({'epoch': epoch})

        # Save the checkpoint
        filepath = os.path.join(args.ckpt_dir, 'ckpt_{:03d}.pth'.format(epoch))
        checkpoint = {
            'backbone': args.model.backbone,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
        }

        if loss < best_loss:
            best_loss = loss
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