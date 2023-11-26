from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

SMALL_NUM = np.log(1e-45)


class DCLLoss(nn.Module):
    """
    Decoupled Contrastive Loss proposed in https://arxiv.org/pdf/2110.06848.pdf
    weight: the weighting function of the positive sample loss
    temperature: temperature to control the sharpness of the distribution
    """

    def __init__(self, temperature=0.07, weight_fn=None, eps=1e-6):
        super(DCLLoss, self).__init__()
        self.temperature = temperature
        self.weight_fn = weight_fn
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.eps = eps
    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size 
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size ):
            mask[i, batch_size  + i] = 0
            mask[batch_size  + i, i] = 0
        return mask
    def forward(self, z_i, z_j):
        """
        Calculate one way DCL loss
        :param z1: first embedding vector
        :param z2: second embedding vector
        :return: one-way loss
        """
        batch_size = z_i.shape[0]               # B

        N = 2 * batch_size    # N
        mask = self.mask_correlated_samples(batch_size) 
        z = torch.cat((z_i, z_j), dim=0)        # (2B x D)
      
        
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0))   # cosine similarity matrix (2B x 2B)
        # print(sim)
        sim= torch.exp(torch.div(sim, self.temperature))
        
        sim_i_j = torch.diag(sim, batch_size )     # the diagonal between z_i*z_j (B) -> positive
        sim_j_i = torch.diag(sim, -batch_size )
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        
        # logit_pos= torch.exp(torch.div(positive_samples, self.temperature))
        logit_pos=positive_samples
        # print(logit_pos)
        negative_samples = sim[mask].reshape(N, -1) 
        # logit_neg= torch.exp(torch.div(negative_samples, self.temperature))
        logit_neg=negative_samples
        sum_neg = torch.sum(logit_neg, dim=1, keepdim=True) + self.eps
        
        # log_pos = torch.log(logit_pos)                                 
        # log_neg = torch.log(sum_neg)
        # loss = torch.sum(-log_pos + log_neg)/2*N


        logit= torch.div(logit_pos, sum_neg + self.eps)
        log= torch.log(logit)
        loss = torch.sum(-log)/N
        return loss


# class DCLW(DCL):
#     """
#     Decoupled Contrastive Loss with negative von Mises-Fisher weighting proposed in https://arxiv.org/pdf/2110.06848.pdf
#     sigma: the weighting function of the positive sample loss
#     temperature: temperature to control the sharpness of the distribution
#     """
#     def __init__(self, sigma=0.5, temperature=0.1):
#         weight_fn = lambda z1, z2: 2 - z1.size(0) * torch.nn.functional.softmax((z1 * z2).sum(dim=1) / sigma, dim=0).squeeze()
#         super(DCLW, self).__init__(weight_fn=weight_fn, temperature=temperature)