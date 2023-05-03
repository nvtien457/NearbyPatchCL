from .moco_aug import MoCoTransform, GaussianBlur
from .simclr_aug import SimCLRTransform
from .byol_aug import BYOLTransform
from .supcon_aug import SupConTransform
from .Multi_FixTransform import Multi_Fixtransform

import torchvision.transforms as T

imagenet_mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return q, k

class Transform_single:
    def __init__(self, image_size, train, normalize=imagenet_mean_std):
        # self.denormalize = Denormalize(*imagenet_norm)
        if train == True:
            self.transform = T.Compose([
                # transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=Image.BICUBIC),
                T.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3.0/4.0,4.0/3.0), interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize(*normalize)
            ])
        else:
            self.transform = T.Compose([
                # transforms.Resize(int(image_size*(8/7)), interpolation=Image.BICUBIC), # 224 -> 256
                T.Resize(int(image_size*(8/7)), interpolation=T.InterpolationMode.BICUBIC), # 224 -> 256  
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(*normalize)
            ])

    def __call__(self, x):
        return self.transform(x)

def get_aug(aug_cfg, train=True):
    name = aug_cfg.name

    if train==True:
        if name == 'moco':
            augmentation = TwoCropsTransform(MoCoTransform(**aug_cfg.params))

        elif name == 'simclr':
            augmentation = TwoCropsTransform(SimCLRTransform(**aug_cfg.params))

        elif name == 'byol':
            augmentation = TwoCropsTransform(BYOLTransform(**aug_cfg.params))

        elif name == 'supcon':
            augmentation = TwoCropsTransform(SupConTransform(**aug_cfg.params))

        elif name == 'clsa':
            augmentation = Multi_Fixtransform(**aug_cfg.params)

        else:
            raise NotImplementedError


    elif train==False:
        # if train_classifier is None:
        #     raise Exception
            
        augmentation = Transform_single(image_size=128, train=False)

    else:
        raise Exception
    
    return augmentation








