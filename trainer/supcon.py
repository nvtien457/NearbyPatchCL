   import torch
import torch.nn.functional as F
import numpy as np

from optimizers import warmup_learning_rate, adjust_learning_rate
from tools import accuracy
import random

def supcon_train(inputs, labels, model, criterion, args):
    batch_size = labels.shape[0]
    n = len(inputs[1])

    # aug of center
    img_0 = torch.cat((inputs[0][0], torch.cat([inputs[1][i][0] for i in range(len(inputs[1]))], dim=0)), dim=0).to(args.device)
    # aug of nearby
    img_1 = torch.cat((inputs[0][1], torch.cat([inputs[1][i][1] for i in range(len(inputs[1]))], dim=0)), dim=0).to(args.device)
    labels = torch.from_numpy(np.tile(np.array([i for i in range(labels.shape[0])]), n+1)).to(args.device)

    index = [i for i in range(labels.shape[0])]
    # random.shuffle(index)

    # compute output
    f0 = model(img_0)
    f1 = model(img_1)
    features = torch.cat([f0.unsqueeze(1), f1.unsqueeze(1)], dim=1)
    
    # features =  f1.unsqueeze(1)
    loss = criterion(features[index], labels[index])

    result_dict = {
        'loss': loss
    }

    logit = torch.mm(f0[:batch_size], f1[:batch_size].T)
    target = torch.arange(0, batch_size, dtype=torch.long).to(args.device)

    for m in args.train.metrics:
        if m == 'acc_1':
            result_dict[m] = accuracy(logit, target, topk=(1,))[0]
        elif m == 'acc_5':
            result_dict[m] = accuracy(logit, target, topk=(5,))[0]

    return result_dict