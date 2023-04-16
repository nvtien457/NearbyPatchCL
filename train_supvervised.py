import torch

from .datasets import get_dataset
from .models import get_model
from .losses import get_criterion

def main():
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

    model = get_model(model_cfg=args.model)

    criterion = get_criterion(criterion_cfg=args.train.criterion)
    
    optimizer = get_optimizer(optimizer_cfg=args.train.optimizer, model=model)
    scheduler = get_scheduler(scheduler_cfg=args.train.scheduler, optimizer=optimizer)

    start_epoch = 0
    best_loss = 9999999

    # Start training
    global_progress = tqdm(range(start_epoch, args.train.stop_epoch), initial=start_epoch, total=args.train.stop_epoch-1, desc='Training')

    trainer = Trainer(train_loader=train_loader, model=model, scaler=scaler,
                    criterion=criterion, optimizer=optimizer, scheduler=scheduler, 
                    logger=logger, args=args)

    for epoch in global_progress:

        # Training
        metrics = trainer.train(epoch)
        loss = metrics['loss_avg']
        
        # if args.train.knn_monitor and (epoch + 1) % args.train.knn_interval == 0:
        #     accuracy = knn_monitor(model.backbone, memory_loader, val_loader, args.device,
        #                     k=min(args.train.knn_k, len(memory_loader.dataset)),
        #                     hide_progress=args.hide_progress)
            
        #     metrics['knn_acc'] = accuracy

        # Display
        global_progress.set_postfix(metrics)
        # logger.update_scalers({'epoch': epoch})

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
            print('best')
            # best_loss = loss
            # filepath = filepath.replace('ckpt_', 'ckpt_best_')
            # torch.save(checkpoint, filepath)

        elif (epoch + 1) % args.train.save_interval == 0:
            print('save')
            # torch.save(checkpoint, filepath)

if __name__=='__main__':
    main()