import torch
import torch.nn.functional as F

from optimizers import warmup_learning_rate, adjust_learning_rate
from tools import accuracy

def supcon_train(inputs, labels, model, criterion, args, epoch, batch_id):
    images = torch.cat([inputs[0], inputs[1]], dim=0)
    batch_size = labels.shape[0]

    images = images.to(device)
    labels = labels.to(device)

    warmup_learning_rate(args=args, optimizer=optimizer, epoch=epoch, batch_id=batch_id, total_batches=)

    # compute output
    z1, p1 = model(img_1)
    z2, p2 = model(img_2)
    z3, p3 = model(img_3)
    loss = criterion(p1, z2) / 2 + criterion(p2, z1) / 2 + criterion(p1, z3) / 2 + criterion(p3, z1) / 2

    result_dict = {
        'loss': loss
    }

    logit = torch.mm(F.normalize(z1, dim=-1), F.normalize(z2, dim=-1).T)
    target = torch.arange(0, batch_size, dtype=torch.long).to(args.device)

    for m in args.train.metrics:
        if m == 'acc@1':
            result_dict[m] = accuracy(logit, target, topk=(1,))[0]
        elif m == 'acc@5':
            result_dict[m] = accuracy(logit, target, topk=(5,))[0]

    return result_dict