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


# class SupConLoss(nn.Module):
#     def __init__(self, temperature=0.07, eps=1e-6, threshold=1):
#         super(SupConLoss, self).__init__()
#         self.temperature = temperature
#         self.similarity_f = nn.CosineSimilarity(dim=2)
#         self.eps = eps
#         self.threshold = threshold

#     def mask_negative(self, batch_size, nearby_size):
#         mask = torch.ones((batch_size*2, batch_size * (2 + nearby_size)), dtype=bool)
#         mask = mask.fill_diagonal_(0)
#         # for i in range(batch_size):
#         #     mask[i, batch_size+i] = 0
#         #     mask[batch_size+i, i] = 0
#         mask[:,2*batch_size:] = 0       # remove nearby from negative samples
#         return mask

#     def mask_positive(self, batch_size, nearby_size):
#         mask = torch.zeros((batch_size*2, batch_size * (2 + nearby_size)), dtype=bool)
#         for i in range(batch_size):
#             mask[i, batch_size+i] = 1
#             mask[batch_size+i, i] = 1
#             for nb in range(nearby_size):
#                 mask[i, 2*batch_size + nb*batch_size + i] = 1
#                 mask[batch_size+i, 2*batch_size + nb*batch_size + i] = 1      
#         return mask

#     def mask_threshold(self, batch_size, nearby_size):
#         mask = torch.ones((batch_size*2, batch_size * (2 + nearby_size)), dtype=bool)

#     def forward(self, z1, z2, zn):
#         batch_size = z1.shape[0]                # B
#         nearby_size = zn.shape[0] // batch_size # N

#         # neg_mask = self.mask_negative(batch_size, nearby_size)    # (2B x (2B-2))
#         # pos_mask = self.mask_positive(batch_size, nearby_size)    # (2B x (2B + N*B))

#         # print(neg_mask)
#         # print('='*24)
#         # print(pos_mask)

#         za = torch.cat((z1, z2), dim=0)        # (2B x D)

#         # sim = self.similarity_f(z.unsqueeze(1), f.unsqueeze(0))     # 2B x (2B + N*B)
#         sim_aug_aug     = self.similarity_f(za.unsqueeze(1), za.unsqueeze(0))   # 2B x 2B
#         sim_aug_nearby  = self.similarity_f(za.unsqueeze(1), zn.unsqueeze(0))   # 2B x N*B

#         # print(sim_aug_aug.shape, sim_aug_nearby.shape)
        
#         thres_mask = torch.ones_like(sim_aug_nearby) * self.threshold           # 2B x N*B
#         sim_aug_nearby = torch.minimum(sim_aug_nearby, thres_mask)

#         # print('sim_aug_aug:', torch.min(sim_aug_aug), torch.max(sim_aug_aug))
#         # print('sim_aug_nearby:', torch.min(sim_aug_nearby), torch.max(sim_aug_nearby))

#         # print('sim:', torch.min(sim), torch.max(sim))
#         # logit = torch.exp(torch.div(sim, self.temperature))  # (2B x (2B + N*B))
#         logit_aug_aug = torch.exp(torch.div(sim_aug_aug, self.temperature))     # 2B x 2B
#         logit_aug_nearby = torch.exp(torch.div(sim_aug_nearby, self.temperature))   # 2B x N*B
#         # print('logit:', torch.min(logit), torch.max(logit))
#         # logit_max, _ = torch.max(logit, dim=1, keepdim=True)
#         # logit_min, _ = torch.min(logit, dim=1, keepdim=True)
#         # logit = (logit) / (logit_max.detach() - logit_min.detach())
#         # print('logit (after):', torch.min(logit), torch.max(logit))

#         # print(logit_aug_aug.shape, logit_aug_nearby.shape)

#         # print('logit_aug_aug:', torch.min(logit_aug_aug), torch.max(logit_aug_aug))
#         # print('logit_aug_nearby:', torch.min(logit_aug_nearby), torch.max(logit_aug_nearby))

#         # positive_samples = logit[pos_mask].reshape(2*batch_size, -1)  # ( 2B x (N+1) )
#         # negative_samples = logit[neg_mask].reshape(2*batch_size, -1)  # ( 2B x (2B-2) )

#         neg_mask = torch.ones_like(logit_aug_aug, dtype=bool).fill_diagonal_(0)
#         negative_samples = logit_aug_aug[neg_mask].reshape(2*batch_size, -1)    # ( 2B x (2B-1) )

#         positive_samples_aug = torch.cat((
#             torch.diag(logit_aug_aug, batch_size),
#             torch.diag(logit_aug_aug, -batch_size)
#         ), dim=0).reshape(2*batch_size, -1)         # ( 2B x 1 )

#         positive_samples_nearby = torch.cat([torch.diag(logit_aug_nearby, i*batch_size) for i in range(-1, nearby_size)]).reshape(2*batch_size, -1)
#                                                     # ( 2B x N )

#         # print(positive_samples_aug.shape, positive_samples_nearby.shape)
#         # print(negative_samples.shape)

#         sum_neg = torch.sum(negative_samples, dim=1, keepdim=True)              # ( 2B x 1 )
#         # print(sum_neg.shape)
#         # print('sum_neg:', torch.min(sum_neg), torch.max(sum_neg))
#         prob_aug_aug = torch.div(positive_samples_aug, sum_neg + self.eps)                  # ( 2B x 1 )
#         prob_aug_nearby = torch.div(positive_samples_nearby, sum_neg + self.eps)            # ( 2B x N )
#         # print(prob_aug_aug.shape, prob_aug_nearby.shape)
#         # print('pos_aug:', torch.min(positive_samples_aug), torch.max(positive_samples_aug))
#         # print('prob_aug:', torch.min(prob_aug_aug), torch.max(prob_aug_aug))
#         # print('pos_nearby:', torch.min(positive_samples_nearby), torch.max(positive_samples_nearby))
#         # print('prob_nearby:', torch.min(prob_aug_nearby), torch.max(prob_aug_nearby))
#         log_prob_aug = -torch.log(prob_aug_aug)                                 # ( 2B x 1 )
#         log_prob_nearby = -torch.log(prob_aug_nearby)                           # ( 2B x N )
#         # print('log_prob_aug:', torch.min(log_prob_aug), torch.max(log_prob_aug))
#         # print('log_prob_nearby:', torch.min(log_prob_nearby), torch.max(log_prob_nearby))
#         mean_log_prob = torch.div(nearby_size * log_prob_aug + torch.sum(log_prob_nearby, dim=1, keepdim=True), 2*nearby_size)  # ( 2B x 1 )
#         # print(torch.sum(log_prob_nearby, dim=1, keepdim=True).shape, mean_log_prob.shape)
#         # print('mean_log_prob:', torch.min(mean_log_prob), torch.max(mean_log_prob))
#         loss = torch.sum(mean_log_prob)

#         loss /= 2*batch_size

#         return loss

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, eps=1e-6, threshold=1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.eps = eps
        self.threshold = threshold

    def neg_mask(self, batch_size):
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
        batch_size = z1.shape[0]                # B
        nearby_size = zn.shape[0] // batch_size # N

        za = torch.cat((z1, z2), dim=0)        # (2B x D)

        # print(za.shape)

        sim_aug_aug     = self.similarity_f(za.unsqueeze(1), za.unsqueeze(0))   # 2B x 2B
        sim_aug_nearby  = self.similarity_f(za.unsqueeze(1), zn.unsqueeze(0))   # 2B x N*B
        thres_mask = torch.ones_like(sim_aug_nearby) * self.threshold           # 2B x N*B
        sim_aug_nearby = torch.minimum(sim_aug_nearby, thres_mask)

        # print(torch.min(sim_aug_aug), torch.max(sim_aug_aug))
        # print(torch.min(sim_aug_nearby), torch.max(sim_aug_nearby))

        logit_aug_aug = torch.exp(torch.div(sim_aug_aug, self.temperature))     # 2B x 2B
        logit_aug_nearby = torch.exp(torch.div(sim_aug_nearby, self.temperature))   # 2B x N*B

        # print(torch.min(logit_aug_aug), torch.max(logit_aug_aug))
        # print(torch.min(logit_aug_nearby), torch.max(logit_aug_nearby))

        neg_mask = torch.ones_like(logit_aug_aug, dtype=bool).fill_diagonal_(0)
        # for i in range(batch_size):           # cmt 2 đường chéo phụ -> 2B-1
        #     neg_mask[i, batch_size+i] = 0
        #     neg_mask[batch_size+i, i] = 0
        negative_samples = logit_aug_aug[neg_mask].reshape(2*batch_size, -1)    # ( 2B x (2B-2) )

        positive_samples_aug = torch.cat((
            torch.diag(logit_aug_aug, batch_size),
            torch.diag(logit_aug_aug, -batch_size)
        ), dim=0).reshape(2*batch_size, -1)         # ( 2B x 1 )

        positive_samples_nearby = torch.cat([torch.diag(logit_aug_nearby, i*batch_size) for i in range(-1, nearby_size)]).reshape(2*batch_size, -1)
                                                    # ( 2B x N )

        # print(positive_samples_aug.shape, positive_samples_nearby.shape, negative_samples.shape)

        sum_neg = torch.sum(negative_samples, dim=1, keepdim=True)  # (2Bx1)

        # print(sum_neg.shape)

        prob_aug_aug = torch.div(positive_samples_aug, sum_neg + self.eps)                  # ( 2B x 1 )
        prob_aug_nearby = torch.div(positive_samples_nearby, sum_neg + self.eps)            # ( 2B x N )

        # print(prob_aug_aug.shape)
        # print(prob_aug_nearby.shape)
        
        log_prob_aug = -torch.log(prob_aug_aug)                                 # ( 2B x 1 )
        log_prob_nearby = -torch.log(prob_aug_nearby)                           # ( 2B x N )

        # print(torch.min(log_prob_aug), torch.max(log_prob_aug))
        # print(torch.min(log_prob_nearby), torch.max(log_prob_nearby))

        mean_log_prob = torch.div(nearby_size * log_prob_aug + torch.sum(log_prob_nearby, dim=1, keepdim=True), 2*nearby_size)  # ( 2B x 1 )

        # print(mean_log_prob.shape)
        # print(torch.min(mean_log_prob), torch.max(mean_log_prob))

        loss = torch.sum(mean_log_prob)                       

        loss /= 2*batch_size

        return loss