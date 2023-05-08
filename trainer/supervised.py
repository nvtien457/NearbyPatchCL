import torch
import torch.nn.functional as F

from tools import accuracy

def supcon_train(inputs, labels, model, criterion, args):
    batch_size = labels.shape[0]

    inputs = inputs.to(args.device)
    labels = inputs.to(args.device)

    # compute output
    preds = model(inputs)
    loss = criterion(preds, labels)

    result_dict = {
        'loss': loss
    }

    for m in args.train.metrics:
        if m == 'acc_1':
            preds = torch.argmax(torch.softmax(preds, dim=1), dim=1)
            result_dict[m] = torch.sum(torch.eq(preds, labels))

    return result_dict