import torch
import torch.nn.functional as F

from optimizers import warmup_learning_rate, adjust_learning_rate
from tools import accuracy

def barlowtwins_nearby_train(inputs, labels, model, criterion, args):
    batch_size = labels.shape[0]

    x_i = inputs[0][0]
    x_j = inputs[0][1]

    nearby = inputs[1]
    x_n = torch.cat(nearby, dim=0)

    x_i = x_i.to(args.device)
    x_j = x_j.to(args.device)
    x_n = x_n.to(args.device)
    # compute output
    z_i, z_j, loss = model(x_i, x_j,x_n)
    

    result_dict = {
        'loss': loss,
    }

    logit = torch.mm(F.normalize(z_i, dim=-1), F.normalize(z_j, dim=-1).T)
    target = torch.arange(0, batch_size, dtype=torch.long).to(args.device)

    for m in args.train.metrics:
        if m == 'acc_1':
            result_dict[m] = accuracy(logit, target, topk=(1,))[0]
        elif m == 'acc_5':
            result_dict[m] = accuracy(logit, target, topk=(5,))[0]
    
    return result_dict