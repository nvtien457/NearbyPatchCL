import torchvision.models as models
import torch

from .moco import MoCo
from .simclr import SimCLR
from .simsiam import SimSiam
from .simtriplet import SimTriplet
from .byol import BYOL
from .supcon import SupCon
from .CLSA import CLSA
# from .supervised import Supervised
from .barlow_twins import BarlowTwins
from .barlow_twins_nearby import BarlowTwins_nearby

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

    elif model_cfg.name == 'supcon':
        model = SupCon(get_backbone(backbone=model_cfg.backbone), **model_cfg.params)

    elif model_cfg.name == 'barlow_twins':
        model = BarlowTwins(backbone=get_backbone(backbone=model_cfg.backbone), **model_cfg.params)
        
    elif model_cfg.name == 'barlow_twins_nearby':
        model = BarlowTwins_nearby(backbone=get_backbone(backbone=model_cfg.backbone), **model_cfg.params)
    elif model_cfg.name == 'clsa':
        model = CLSA(backbone=get_backbone(backbone=model_cfg.backbone), **model_cfg.params)

    # elif model_cfg.name == 'supervised':
    #     model = Supervised(backbone=get_backbone(backbone=model_cfg.backbone), **model_cfg.params)

    elif model_cfg.name == 'swav':
        raise NotImplementedError

    else:
        raise NotImplementedError

    return model