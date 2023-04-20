from tools import accuracy

def moco_train(inputs, labels, model, criterion, args):
    queries = inputs[0]
    keys    = inputs[1]

    queries = queries.to(args.device)
    keys    = keys.to(args.device)

    # compute output
    output, target = model(im_q=queries, im_k=keys)
    loss = criterion(output, target)

    result_dict = {
        'loss': loss
    }

    for m in args.train.metrics:
        if m == 'acc_1':
            result_dict[m] = accuracy(output, target, topk=(1,))[0]
        elif m == 'acc_5':
            result_dict[m] = accuracy(output, target, topk=(5,))[0]

    return result_dict