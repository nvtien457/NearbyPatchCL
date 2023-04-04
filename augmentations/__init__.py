from .moco_aug import MoCoTransform, TwoCropsTransform, GaussianBlur

def get_aug(name='simsiam', image_size=224, train=True, train_classifier=None, version=None):
    if train==True:
        if name == 'moco':
            augmentation = TwoCropsTransform(MoCoTransform(image_size, version=version))
        else:
            raise NotImplementedError
    elif train==False:
        if train_classifier is None:
            raise Exception
        augmentation = Transform_single(image_size, train=train_classifier)
    else:
        raise Exception
    
    return augmentation








