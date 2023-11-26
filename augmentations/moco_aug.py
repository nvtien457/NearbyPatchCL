'''
Modify from https://github.com/facebookresearch/moco-v3
'''

from PIL import ImageFilter
import random
import torchvision.transforms as T


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])

class MoCoTransform:
    def __init__(self, image_size=224, normalize=normalize, version='v1'):
        assert version in ['v1', 'v2']
        self.version = version

        if version == 'v1':
            # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978      
            self.transform = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.2, 1.)),
                T.RandomGrayscale(p=0.2),
                T.ColorJitter(0.4, 0.4, 0.4, 0.4),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                normalize
            ])

        else:
            # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
            self.transform = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.2, 1.)),
                T.RandomApply([
                    T.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                normalize
            ])

    def __call__(self, x):
        return self.transform(x)