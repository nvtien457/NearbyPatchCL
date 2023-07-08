import torch
import torch.nn.functional as F

from optimizers import warmup_learning_rate, adjust_learning_rate
from tools import accuracy
import random

def micle_train(inputs, labels, model, criterion, args):
    batch_size = labels.shape[0]
    n = len(inputs[1])

    img = torch.cat((inputs[0], torch.cat([inputs[1][i] for i in range(len(inputs[1]))], dim=0)), dim=0).to(args.device)
    labels = torch.Tensor([[i for _ in range(n+1)] for i in range(batch_size)]).flatten().to(args.device)

    index = [i for i in range(labels.shape[0])]
    random.shuffle(index)

    # compute output
    features = model(img)

    loss = criterion(features[index].unsqueeze(1), labels[index])

    result_dict = {
        'loss': loss
    }

    logit = torch.mm(features[:batch_size], features[batch_size:2*batch_size].T)
    target = torch.arange(0, batch_size, dtype=torch.long).to(args.device)

    for m in args.train.metrics:
        if m == 'acc_1':
            result_dict[m] = accuracy(logit, target, topk=(1,))[0]
        elif m == 'acc_5':
            result_dict[m] = accuracy(logit, target, topk=(5,))[0]

    return result_dict