from datasets.folder_dataset import ImageFolder
from tools import get_args, AverageMeter, Logger
from augmentations import get_aug
from models import get_model, get_backbone
from datasets import get_dataset
from optimizers import get_optimizer, get_scheduler

import os
import shutil
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.model_selection import KFold
from tqdm import tqdm
from torchvision import datasets
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset, SubsetRandomSampler
import torch.utils.data as data
import csv
import pathlib


def main(args):
    # create log & ckpt
    dt = '_' + datetime.now().strftime('%m%d%H%M%S')
    args.log_dir = os.path.join(args.log_dir, 'in-progress_' + args.name + '_finetune' + dt)
    args.ckpt_dir = os.path.join(args.ckpt_dir, args.name + '/finetune' + dt)
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

    dataset = ImageFolder(root=args.val_dir, transform=get_aug(args.aug, train=False))
    splits = KFold(n_splits=args.eval.fold, shuffle=True, random_state=args.seed)
    
    train_info = []
    best_epoch = np.zeros(5)
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
        best_balance_acc = 0

        print('---------------------------------------------------------')
        print('>> FOLD', fold)
        
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=train_sampler,
            batch_size=args.eval.batch_size,
            # shuffle=True,
            **args.dataloader_kwargs
        )
        
        test_loader = torch.utils.data.DataLoader(
            dataset= dataset,
            sampler=test_sampler,
            batch_size=args.eval.batch_size,
            # shuffle=False,
            **args.dataloader_kwargs
        )

        num_classes = len(dataset.classes)

        backbone = get_backbone(args.model.backbone)

        assert args.eval.eval_from is not None
        save_dict = torch.load(args.eval.eval_from, map_location='cpu')
        msg = backbone.load_state_dict({k[9:]: v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')},
                                    strict=True)

        # # print(msg)
        model = backbone.to(args.device)
        model.eval()

        classifier = nn.Linear(in_features=backbone.output_dim, out_features=num_classes, bias=True).to(args.device)

        # Use Mixed Precision Training
        args.eval.optimizer.params['lr']        = args.eval.optimizer.params['lr']        * args.eval.batch_size / 256
        args.eval.scheduler.params['base_lr']   = args.eval.scheduler.params['base_lr']   * args.eval.batch_size / 256
        args.eval.scheduler.params['warmup_lr'] = args.eval.scheduler.params['warmup_lr'] * args.eval.batch_size / 256
        args.eval.scheduler.params['final_lr']  = args.eval.scheduler.params['final_lr']  * args.eval.batch_size / 256

        # Number of iteration per epoch
        iter_per_epoch = len(train_loader)
        args.eval.scheduler.params['iter_per_epoch'] = iter_per_epoch

        optimizer = get_optimizer(optimizer_cfg=args.eval.optimizer, model=classifier)
        scheduler = get_scheduler(scheduler_cfg=args.eval.scheduler, optimizer=optimizer)

        loss_meter = AverageMeter(name='Loss')
        acc_meter = AverageMeter(name='Accuracy')

        logger = Logger(log_dir=args.log_dir, tensorboard=args.tensorboard, matplotlib=args.matplotlib, file=f'plotter_fold{fold}.svg')

        # Start training
        best_ckpt_path = f'{args.ckpt_dir}/fold_{fold}.pth'
        global_progress = tqdm(range(0, args.eval.num_epochs), desc=f'Evaluating')
        for epoch in global_progress:
            loss_meter.reset()
            acc_meter.reset()
            classifier.train()
            local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.eval.num_epochs}')

            for idx, (images, labels) in enumerate(local_progress):
                classifier.zero_grad()
                with torch.no_grad():
                    feature = model(images.to(args.device))

                preds = classifier(feature)

                loss = F.cross_entropy(preds, labels.to(args.device))

                loss.backward()
                optimizer.step()
                lr = scheduler.step()

                with torch.no_grad():
                    preds = preds.argmax(dim=1)
                    correct = (preds == labels.to(args.device)).sum().item()
                    acc_meter.update(correct / preds.shape[0], n=preds.shape[0])

                loss_meter.update(loss.item(), n=preds.shape[0])
                logger.update_scalers({
                    'loss': loss_meter.val,
                    'lr': lr
                })
                local_progress.set_postfix({'lr': lr, "train_loss": loss_meter.val, 'train_loss_avg': loss_meter.avg, 'train_acc': acc_meter.avg * 100})

            logger.update_scalers({
                'train_loss': loss_meter.avg,
                'train_acc': acc_meter.avg * 100,
            })

            writer.add_scalar('Valid/Loss', loss_meter.avg, epoch)
            writer.add_scalar('Valid/Lr', lr, epoch)
            writer.flush()

            fold_path = f'{args.ckpt_dir}/fold_{fold}'

            if not os.path.exists(fold_path):
                os.mkdir(fold_path)

            model_path = f'{fold_path}/f{fold}_tunelinear_{str(epoch)}.pth'

            torch.save(classifier, model_path)

            classifier.eval()
            correct, total = 0, 0
            loss_meter.reset()
            acc_meter.reset()

            pred_label_for_f1 = np.array([])
            true_label_for_f1 = np.array([])
            for idx, (images, labels) in enumerate(test_loader):
                with torch.no_grad():
                    feature = model(images.to(args.device))
                    preds = classifier(feature)
                    loss = F.cross_entropy(preds, labels.to(args.device))
                    loss_meter.update(loss.item(), n=preds.shape[0])
                    preds = preds.argmax(dim=1)
                    correct = (preds == labels.to(args.device)).sum().item()

                    preds_arr = preds.cpu().detach().numpy()
                    labels_arr = labels.cpu().detach().numpy()
                    pred_label_for_f1 = np.concatenate([pred_label_for_f1, preds_arr])
                    true_label_for_f1 = np.concatenate([true_label_for_f1, labels_arr])
                    acc_meter.update(correct / preds.shape[0], n=preds.shape[0])

            f1 = f1_score(true_label_for_f1, pred_label_for_f1, average='macro')
            balance_acc = balanced_accuracy_score(true_label_for_f1, pred_label_for_f1)
            global_progress.set_postfix({'Epoch': epoch, 'Accuracy': acc_meter.avg * 100, 'F1-score': f1, 'Balanced_accuracy': balance_acc})
            logger.update_scalers({
                'val_loss': loss_meter.avg,
                'val_acc': acc_meter.avg * 100,
                'f1': f1,
                'balanced_acc': balance_acc * 100
            })

            if balance_acc > best_balance_acc:
                best_epoch[fold] = epoch
                best_balance_acc = balance_acc
                shutil.copy2(src=model_path, dst=best_ckpt_path)
            train_info.append([epoch, f1, balance_acc])

    with open(f'{args.ckpt_dir}/train_info.csv', 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(train_info)

    print('Best epoch at each fold:', best_epoch)

if __name__ == "__main__":
    writer = SummaryWriter()
    args=get_args()
    main(args)

    completed_log_dir = args.log_dir.replace('in-progress', 'debug' if args.debug else 'completed')

    os.rename(args.log_dir, completed_log_dir)
    print(f'Log file has been saved to {completed_log_dir}')