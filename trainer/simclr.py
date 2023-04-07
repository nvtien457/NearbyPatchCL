import torch

from tools import accuracy

def simclr_train(inputs, labels, model, criterion, args):
    batch_size = labels.shape[0]

    x_i = inputs[0]
    x_j = inputs[1]

    x_i = x_i.to(args.device)
    x_j = x_j.to(args.device)

    # compute output
    z_i, p_i = model(x_i)
    z_j, p_j = model(x_j)
    loss = criterion(p_i, p_j)

    result_dict = {
        'loss': loss,
    }

    output = torch.mm(p_i, p_j.T)
    target = torch.arange(0, batch_size, dtype=torch.long).to(args.device)

    for m in args.train.metrics:
        if m == 'acc@1':
            result_dict[m] = accuracy(output, target, topk=(1,))[0]
        elif m == 'acc@5':
            result_dict[m] = accuracy(output, target, topk=(5,))[0]
    
    return result_dict