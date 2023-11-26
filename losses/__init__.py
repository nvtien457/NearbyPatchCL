import torch.nn as nn

from .NT_Xent import NT_Xent
from .neg_cosine import Negative_CosineSimilarity
from .supcon import SupConLoss, ModifySupConLoss, NNSupConLoss
from .focal import FocalLoss
from .DCL import DCLLoss

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

    elif criterion_cfg.name == 'supcon':
        return SupConLoss(**criterion_cfg.params)
    
    elif criterion_cfg.name == 'nnsupcon':
        return NNSupConLoss(**criterion_cfg.params)
    
    elif criterion_cfg.name == 'DCL':
        return DCLLoss(**criterion_cfg.params)

    
    elif criterion_cfg.name == 'supcon-modify':
        return ModifySupConLoss(**criterion_cfg.params)

    elif criterion_cfg.name == 'focal':
        return FocalLoss(**criterion_cfg.params)

    else:
        raise NotImplementedError