import torchvision.models as models
import torch

from .moco import MoCo
from .simclr import SimCLR
from .simsiam import SimSiam
from .simtriplet import SimTriplet
from .byol import BYOL

def get_backbone(backbone, castrate=True):           #lq debug
    backbone = models.__dict__[backbone]()

    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone

def get_model(model_cfg):
    if model_cfg.name == 'moco':
        model = MoCo(get_backbone(model_cfg.backbone), **model_cfg.params)
    
    elif model_cfg.name == 'simclr':
        model = SimCLR(get_backbone(model_cfg.backbone), **model_cfg.params)

    elif model_cfg.name == 'simsiam':
        model = SimSiam(get_backbone(model_cfg.backbone), **model_cfg.params)

    elif model_cfg.name == 'simtriplet':
        model = SimTriplet(get_backbone(backbone=model_cfg.backbone))

    elif model_cfg.name == 'byol':
        model = BYOL(get_backbone(backbone=model_cfg.backbone), **model_cfg.params)

    elif model_cfg.name == 'swav':
        raise NotImplementedError

    else:
        raise NotImplementedError

    return model