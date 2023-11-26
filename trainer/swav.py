import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from optimizers import warmup_learning_rate, adjust_learning_rate
from tools import accuracy
import random

def swav_train(inputs, labels, model, criterion, args):
    batch_size = labels.shape[0]
    nmb_neighbors = len(inputs[1])
    nmb_crops = len(inputs[0])

    # print('\n')
    # print('Number of nearby:', nmb_neighbors)
    # print('Number of crops:', nmb_crops)

    X = []
    for n in range(nmb_crops):
      img_n = [inputs[0][n]] + [inputs[1][i][n] for i in range(len(inputs[1]))]
      # print('img_n:', len(img_n))
      X.extend(img_n)

    # print('X:', len(X))

    # normalize the prototypes
    with torch.no_grad():
        w = model.prototypes.weight.data.clone()
        w = nn.functional.normalize(w, dim=1, p=2)
        model.prototypes.weight.copy_(w)

    # ============ multi-res forward passes ... ============
    embedding, output = model(X)
    embedding = embedding.detach()

    # print('output:', output.shape)
    # print('embedding:', embedding.shape)

    # ============ swav loss ... ============
    loss = criterion(output, nmb_crops=len(X), crops_for_assign=args.train.crops_for_assign, batch_size=batch_size)

    result_dict = {
        'loss': loss
    }

    logit = torch.mm(embedding[:batch_size], embedding[batch_size:2*batch_size].T)
    target = torch.arange(0, batch_size, dtype=torch.long).to(args.device)

    for m in args.train.metrics:
        if m == 'acc_1':
            result_dict[m] = accuracy(logit, target, topk=(1,))[0]
        elif m == 'acc_5':
            result_dict[m] = accuracy(logit, target, topk=(5,))[0]

    return result_dict