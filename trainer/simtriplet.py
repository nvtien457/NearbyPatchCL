import torch
import torch.nn.functional as F

from tools import accuracy

def simtriplet_train(inputs, labels, model, criterion, args):
    img_1 = inputs[0][0]
    img_2 = inputs[0][1]
    img_3 = inputs[1][0]
    batch_size = labels.shape[0]

    img_1 = img_1.to(args.device)
    img_2 = img_2.to(args.device)
    img_3 = img_3.to(args.device)

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