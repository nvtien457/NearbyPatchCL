from .moco_aug import MoCoTransform, TwoCropsTransform, GaussianBlur
from .simclr_aug import SimCLRTransform
from .byol_aug import BYOLTransform

def get_aug(aug_cfg, train=True):
    name = aug_cfg.name

    if train==True:
        if name == 'moco':
            augmentation = TwoCropsTransform(MoCoTransform(**aug_cfg.params))

        elif name == 'simclr':
            augmentation = TwoCropsTransform(SimCLRTransform(**aug_cfg.params))

        elif name == 'byol':
            augmentation = TwoCropsTransform(BYOLTransform(**aug_cfg.params))

        else:
            raise NotImplementedError


    elif train==False:
        if train_classifier is None:
            raise Exception
            
        augmentation = Transform_single(image_size, train=train_classifier)

    else:
        raise Exception
    
    return augmentation








