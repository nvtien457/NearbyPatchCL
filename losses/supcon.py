"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn

# class SupConLoss(nn.Module):
#     """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
#     It also supports the unsupervised contrastive loss in SimCLR"""
#     def __init__(self, temperature=0.07, contrast_mode='all',
#                  base_temperature=0.07):
#         super(SupConLoss, self).__init__()
#         self.temperature = temperature
#         self.contrast_mode = contrast_mode
#         self.base_temperature = base_temperature

#     def forward(self, features, labels=None, mask=None):
#         """Compute loss for model. If both `labels` and `mask` are None,
#         it degenerates to SimCLR unsupervised loss:
#         https://arxiv.org/pdf/2002.05709.pdf
#         Args:
#             features: hidden vector of shape [bsz, n_views, ...].
#             labels: ground truth of shape [bsz].
#             mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
#                 has the same class as sample i. Can be asymmetric.
#         Returns:
#             A loss scalar.
#         """
#         device = (torch.device('cuda')
#                   if features.is_cuda
#                   else torch.device('cpu'))

#         if len(features.shape) < 3:
#             raise ValueError('`features` needs to be [bsz, n_views, ...],'
#                              'at least 3 dimensions are required')
#         if len(features.shape) > 3:
#             features = features.view(features.shape[0], features.shape[1], -1)

#         batch_size = features.shape[0]
#         if labels is not None and mask is not None:
#             raise ValueError('Cannot define both `labels` and `mask`')
#         elif labels is None and mask is None:
#             mask = torch.eye(batch_size, dtype=torch.float32).to(device)
#         elif labels is not None:
#             labels = labels.contiguous().view(-1, 1)
#             if labels.shape[0] != batch_size:
#                 raise ValueError('Num of labels does not match num of features')
#             mask = torch.eq(labels, labels.T).float().to(device)
#         else:
#             mask = mask.float().to(device)

#         contrast_count = features.shape[1]
#         contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
#         if self.contrast_mode == 'one':
#             anchor_feature = features[:, 0]
#             anchor_count = 1
#         elif self.contrast_mode == 'all':
#             anchor_feature = contrast_feature
#             anchor_count = contrast_count
#         else:
#             raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

#         # compute logits
#         anchor_dot_contrast = torch.div(
#             torch.matmul(anchor_feature, contrast_feature.T),
#             self.temperature)
#         # for numerical stability
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         logits = anchor_dot_contrast - logits_max.detach()

#         # tile mask
#         mask = mask.repeat(anchor_count, contrast_count)
#         # mask-out self-contrast cases
#         logits_mask = torch.scatter(
#             torch.ones_like(mask),
#             1,
#             torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
#             0
#         )                               # negative                  (remove diagonal)
#         mask = mask * logits_mask       # positive without anchor   (remove diagonal)

#         # compute log_prob
#         exp_logits = torch.exp(logits) * logits_mask
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # sum of positive

#         # compute mean of log-likelihood over positive
#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

#         # loss
#         loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
#         loss = loss.view(anchor_count, batch_size).mean()

#         return loss


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, eps=1e-6):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.eps = eps

    def mask_correlated_samples(self, batch_size):
        mask = torch.ones((2*batch_size, 3*batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size+i] = 0
            # mask[batch_size+i, 2*batch_size+i] = 0
            # mask[i, 2*batch_size+i] = 0
            mask[batch_size+i, i] = 0
        mask[:,2*batch_size:] = 0
        return mask

    def forward(self, z1, z2, zn):
        batch_size = z1.shape[0]               # B

        mask = self.mask_correlated_samples(batch_size)    # only negative sample (different samples in batch)
        # (2B x 3B)
        # print(mask)

        z = torch.cat((z1, z2), dim=0)        # (2B x D)
        f = torch.cat((z1,z2, zn), dim=0)     # (3B x D)

        sim = self.similarity_f(z.unsqueeze(1), f.unsqueeze(0))
        # print('sim:', torch.min(sim), torch.max(sim))
        logit = torch.exp(torch.div(sim, self.temperature))  # (2B x 3B)
        # print('logit:', torch.min(logit), torch.max(logit))
        # logit_max, _ = torch.max(logit, dim=1, keepdim=True)
        # logit_min, _ = torch.min(logit, dim=1, keepdim=True)
        # logit = (logit) / (logit_max.detach() - logit_min.detach())
        # print('logit (after):', torch.min(logit), torch.max(logit))

        positive_samples = torch.cat((
            torch.diag(logit, batch_size),
            torch.diag(logit, -batch_size),
            torch.diag(logit, 2*batch_size)
        ), dim=0).reshape(2*batch_size, -1)                      # ( 2B x 2 )
        negative_samples = logit[mask].reshape(2*batch_size, -1)  # ( 2B x (2B-2) )

        # print(positive_samples.shape, negative_samples.shape)

        sum_neg = torch.sum(negative_samples, dim=1, keepdim=True)  # (2Bx1)
        # print('sum_neg:', torch.min(sum_neg), torch.max(sum_neg))
        prob = torch.div(positive_samples, sum_neg + self.eps)                 # (2Bx2)
        # print('pos:', torch.min(positive_samples), torch.max(positive_samples))
        # print('prob:', torch.min(prob), torch.max(prob))
        log_prob = -torch.log(prob)                                 # (2Bx4)
        # print('log_prob:', torch.min(log_prob), torch.max(log_prob))
        mean_log_prob = torch.mean(log_prob, dim=1)             # (B)
        # print('mean_log_prob:', torch.min(mean_log_prob), torch.max(mean_log_prob))
        loss = torch.sum(mean_log_prob)                       

        loss /= batch_size

        return loss