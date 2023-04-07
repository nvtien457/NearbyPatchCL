from tools import accuracy

def byol_train(inputs, labels, model, criterion, args):
    img0 = inputs[0]
    img1 = inputs[1]

    (z0, p0), (z1, p1) = model(img0, img1)

    losses = 2 * (criterion(p0, z1) + criterion(p1, z0))

    result_dict = {
        'loss': loss
    }

    for m in args.train.metrics:
        if m == 'acc@1':
            result_dict[m] = accuracy(output, target, topk=(1,))[0]
        elif m == 'acc@5':
            result_dict[m] = accuracy(output, target, topk=(5,))[0]

    return result_dict