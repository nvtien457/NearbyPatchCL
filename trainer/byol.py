import torch
import torch.nn.functional as F

from tools import accuracy

def byol_train(inputs, labels, model, criterion, args):
    batch_size = labels.shape[0]

    img0 = inputs[0][0]
    img1 = inputs[0][1]

    img0 = img0.to(args.device)
    img1 = img1.to(args.device)

    (z0, p0), (z1, p1) = model(img0, img1)

    loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0))

    result_dict = {
        'loss': loss
    }

    logit = torch.mm(F.normalize(z0, dim=-1), F.normalize(z1, dim=-1).T)
    target = torch.arange(0, batch_size, dtype=torch.long).to(args.device)

    for m in args.train.metrics:
        if m == 'acc_1':
            result_dict[m] = accuracy(logit, target, topk=(1,))[0]
        elif m == 'acc_5':
            result_dict[m] = accuracy(logit, target, topk=(5,))[0]

    return result_dict