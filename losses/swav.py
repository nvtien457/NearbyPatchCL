from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class SwAVLoss(nn.Module):
    def __init__(
        self, 
        temperature=0.1, 
        epsilon=0.5,
        sinkhorn_iterations=3
    ):
        super(SwAVLoss, self).__init__()
        self.temperature = temperature
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations

    def forward(self, output, nmb_crops, crops_for_assign, batch_size):
        loss = 0
        for i, crop_id in enumerate(crops_for_assign):
            with torch.no_grad():
                out = output[batch_size * crop_id: batch_size * (crop_id + 1)].detach()

                # # time to use the queue
                # if queue is not None:
                #     if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                #         use_the_queue = True
                #         out = torch.cat((torch.mm(
                #             queue[i],
                #             model.module.prototypes.weight.t()
                #         ), out))
                #     # fill the queue
                #     queue[i, bs:] = queue[i, :-bs].clone()
                #     queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                # get assignments
                q = sinkhorn(out, self.epsilon, self.sinkhorn_iterations)[-batch_size:]

            # cluster assignment prediction
            subloss = 0
            # print(nmb_crops)
            # print(crop_id)
            for v in np.delete(np.arange(np.sum(nmb_crops)), crop_id):
                x = output[batch_size * v: batch_size * (v + 1)] / self.temperature
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            loss += subloss / (np.sum(nmb_crops) - 1)
        loss /= len(crops_for_assign)
        return loss

@torch.no_grad()
def sinkhorn(out, epsilon, sinkhorn_iterations):
    Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] # number of samples to assign
    K = Q.shape[0] # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()