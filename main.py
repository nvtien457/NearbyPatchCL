from tqdm import tqdm

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
    args.train.optimizer.params['lr']        = args.train.optimizer.params['lr']        * args.train.batch_size / 256
    args.train.scheduler.params['base_lr']   = args.train.scheduler.params['base_lr']   * args.train.batch_size / 256
    args.train.scheduler.params['warmup_lr'] = args.train.scheduler.params['warmup_lr'] * args.train.batch_size / 256
    args.train.scheduler.params['final_lr']  = args.train.scheduler.params['final_lr']  * args.train.batch_size / 256

    # Number of iteration per epoch
    iter_per_epoch = len(train_loader)
    args.train.scheduler.params['iter_per_epoch'] = iter_per_epoch

    optimizer = get_optimizer(optimizer_cfg=args.train.optimizer, model=model)
    scheduler = get_scheduler(scheduler_cfg=args.train.scheduler, optimizer=optimizer)

    best_loss = 99999999
    global_progress = tqdm(range(args.train.num_epochs), desc='Training')
    logger = Logger(log_dir=args.log_dir, tensorboard=args.tensorboard, matplotlib=args.matplotlib)

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

        # Save the checkpoint
        filename = 'ckpt_{:03d}.pth'.format(epoch)
        checkpoint = {
            'backbone': args.model.backbone,
            'state_dict': model.state_dict(),
            'optimizers': optimizer.state_dict(),
            'epoch': epoch,
        }

        if loss < best_loss:
            best_loss = loss
            filename.replace('ckpt_', 'ckpt_best_')
            # torch.save(checkpoint, filename)
            print('best')

        elif (epoch + 1) % args.train.save_interval == 0:
            # torch.save(checkpoint, filename)
            print('save')


if __name__=='__main__':
    args = get_args()

    main(args)