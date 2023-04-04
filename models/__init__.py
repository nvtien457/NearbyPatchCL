from torchvision.models import resnet50, resnet18
import torchvision.models as models
import torch

from .moco import MoCo

def get_backbone(backbone, castrate=True):           #lq debug
    backbone = eval(f"{backbone}()")

    if castrate:
        backbone.output_dim = backbone.fc.in_features
        backbone.fc = torch.nn.Identity()

    return backbone

def get_model(model_cfg, castrate=False):
    if model_cfg.name == 'moco':
        model = MoCo(base_encoder=models.__dict__[model_cfg.backbone], 
                    dim=model_cfg.dim, K=model_cfg.K, m=model_cfg.m, T=model_cfg.T, mlp=model_cfg.mlp)

    elif model_cfg.name == 'swav':
        raise NotImplementedError

    else:
        raise NotImplementedError

    return model