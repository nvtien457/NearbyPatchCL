from tools import accuracy
import torch

def simsiam_train(inputs, labels, model, criterion, args):
    img_1 = inputs[0]
    img_2 = inputs[1]
    batch_size = labels.shape[0]

    img_1 = img_1.to(args.device)
    img_2 = img_2.to(args.device)

    # compute output
    z1, p1 = model(img_1)
    z2, p2 = model(img_2)
    loss = (criterion(p1, z2) + criterion(p2, z1)) * 0.5

    result_dict = {
        'loss': loss
    }
    
    output = torch.mm(p1, p2.T)
    target = torch.arange(0, batch_size, dtype=torch.long).to(args.device)

    for m in args.train.metrics:
        if m == 'acc@1':
            result_dict[m] = accuracy(output, target, topk=(1,))[0]
        elif m == 'acc@5':
            result_dict[m] = accuracy(output, target, topk=(5,))[0]

    return result_dict