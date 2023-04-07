import torch.nn as nn

from .NT_Xent import NT_Xent
from .neg_cosine import Negative_CosineSimilarity

def get_criterion(criterion_cfg):
    if criterion_cfg.name == 'CE':
        return nn.CrossEntropyLoss(**criterion_cfg.params)

    elif criterion_cfg.name == 'NT-Xent':
        return NT_Xent(**criterion_cfg.params)

    elif criterion_cfg.name == 'MSE':
        return nn.MSELoss(**criterion_cfg.params)

    elif criterion_cfg.name == 'cosine':
        return nn.CosineSimilarity(**criterion_cfg.params)

    elif criterion_cfg.name == 'neg-cosine':
        return Negative_CosineSimilarity

    else:
        raise NotImplementedError