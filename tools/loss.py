import torch
import torch.nn.functional as F

def nce_loss(feature1, feature2, temp=0.07):
    feature1 = F.normalize(feature1, dim=1)
    feature2 = F.normalize(feature2, dim=1)

    # print(torch.max(feature1).item(), torch.min(feature1).item())
    # print(torch.max(feature2).item(), torch.min(feature2).item())

    logit = torch.exp(torch.mm(feature1, feature2.T) / temp)    # N x N

    # print(torch.max(logit).item(), torch.min(logit).item())

    positive = torch.diagonal(logit)    # N
    negative = torch.sum(logit, dim=0) + torch.sum(logit, dim=1) - 2*positive # N

    return torch.mean(-torch.log(positive / negative))