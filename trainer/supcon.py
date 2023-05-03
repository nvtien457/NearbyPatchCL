import torch
import torch.nn.functional as F

from optimizers import warmup_learning_rate, adjust_learning_rate
from tools import accuracy

def supcon_train(inputs, labels, model, criterion, args):
    batch_size = labels.shape[0]

    img_1 = inputs[0][0]
    img_2 = inputs[0][1]
    nearby = inputs[1]
    img_3 = torch.cat(nearby, dim=0)

    img_1 = img_1.to(args.device)
    img_2 = img_2.to(args.device)
    img_3 = img_3.to(args.device)

    # compute output
    p1 = model(img_1)
    p2 = model(img_2)
    p3 = model(img_3)
    loss = criterion(p1, p2, p3)

    result_dict = {
        'loss': loss
    }

    logit = torch.mm(p1, p2.T)
    target = torch.arange(0, batch_size, dtype=torch.long).to(args.device)

    for m in args.train.metrics:
        if m == 'acc_1':
            result_dict[m] = accuracy(logit, target, topk=(1,))[0]
        elif m == 'acc_5':
            result_dict[m] = accuracy(logit, target, topk=(5,))[0]

    return result_dict