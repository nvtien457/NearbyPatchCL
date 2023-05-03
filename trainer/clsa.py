from tools import accuracy
import torch

def clsa_train(inputs, labels, model, criterion, args):
    len_inputs = len(inputs)

    for k in range(len(inputs)):
        inputs[k] = inputs[k].to(args.device, non_blocking=True)
    crop_copy_length = int((len_inputs - 1) / 2)
    image_k = inputs[0]
    image_q = inputs[1:1 + crop_copy_length]
    image_strong = inputs[1 + crop_copy_length:]

    # compute output
    output, target, output2, target2 = model(image_q, image_k, image_strong)
    
    loss_contrastive = 0
    loss_weak_strong = 0
    for k in range(len(output)):
        loss1 = criterion(output[k], target[k])
        loss_contrastive += loss1
    for k in range(len(output2)):
        loss2 = -torch.mean(torch.sum(torch.log(output2[k]) * target2[k], dim=1))  # DDM loss
        loss_weak_strong += loss2

    loss = loss_contrastive + args.train.alpha * loss_weak_strong

    result_dict = {
        'loss': loss
    }

    for m in args.train.metrics:
        if m == 'acc_1':
            result_dict[m] = accuracy(output[0], target[0], topk=(1,))[0]
        elif m == 'acc_5':
            result_dict[m] = accuracy(output[0], target[0], topk=(5,))[0]

    return result_dict