#modified from https://github.com/facebookresearch/swav/blob/master/src/multicropdataset.py
import torchvision.transforms as T
from augmentations.RandAugment import RandAugment
import random
from PIL import ImageFilter

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

imagenet_mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

class Multi_Fixtransform(object):
    def __init__(self,
            size_crops,
            nmb_crops,
            min_scale_crops,
            max_scale_crops,
            aug_times,
            normalize=T.Normalize(*imagenet_mean_std),
            init_size=128):
        """
        :param size_crops: list of crops with crop output img size
        :param nmb_crops: number of output cropped image
        :param min_scale_crops: minimum scale for corresponding crop
        :param max_scale_crops: maximum scale for corresponding crop
        :param normalize: normalize operation
        :param aug_times: strong augmentation times
        :param init_size: key image size
        """
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        trans=[]
        #key image transform
        self.weak = T.Compose([
            T.RandomResizedCrop(init_size, scale=(0.2, 1.)),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
        trans.append(self.weak)
        self.aug_times=aug_times
        
        trans_weak=[]
        trans_strong=[]
        for i in range(len(size_crops)):

            randomresizedcrop = T.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )

            strong = T.Compose([
                randomresizedcrop,
                T.RandomApply([
                    T.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                T.RandomHorizontalFlip(),
                RandAugment(n=self.aug_times, m=10),
                T.ToTensor(),
                normalize
            ])

            weak = T.Compose([
                randomresizedcrop,
                T.RandomApply([
                    T.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])

            trans_weak.extend([weak]*nmb_crops[i])
            trans_strong.extend([strong]*nmb_crops[i])


        trans.extend(trans_weak)
        trans.extend(trans_strong)
        self.trans=trans


    def __call__(self, x):
        multi_crops = list(map(lambda trans: trans(x), self.trans))
        return multi_crops