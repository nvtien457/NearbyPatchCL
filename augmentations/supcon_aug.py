'''
Modify from https://github.com/HobbitLong/SupContrast
'''

import torchvision.transforms as T

class SupConTransform:
    def __init__(self, image_size=224, dataset='imagenet'):
        assert dataset in ['cifar10', 'cifar100', 'imagenet']
        if dataset == 'cifar10':
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
        elif dataset == 'cifar100':
            mean = (0.5071, 0.4867, 0.4408)
            std = (0.2675, 0.2565, 0.2761)
        elif dataset == 'imagenet':
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        else:
            raise ValueError('dataset not supported: {}'.format(dataset))
        normalize = T.Normalize(mean=mean, std=std)

        self.transform = T.Compose(
            [
                T.RandomResizedCrop(size=image_size, scale=(0.2, 1.)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomApply([
                    T.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                normalize
            ]
        )

    def __call__(self, x):
        return self.transform(x)