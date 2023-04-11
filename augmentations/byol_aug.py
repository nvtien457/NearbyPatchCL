import torchvision

imagenet_mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

class BYOLTransform:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size, train=True, mean_std=imagenet_mean_std):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = torchvision.transforms.Compose(
            [
                T.RandomResizedCrop(size=size),
                T.RandomHorizontalFlip(),  # with 0.5 probability
                T.RandomVerticalFlip(),
                T.RandomApply([color_jitter], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.GaussianBlur(kernel_size=23),
                T.RandomSolarize(threshold=0.5),
                T.ToTensor(),
                T.Normalize(*mean_std)
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [
                T.Resize(size=size),
                T.ToTensor(),
            ]
        )

        if train:
            self.transform = self.train_transform
        else:
            self.transform = self.test_transform

    def __call__(self, x):
        return self.transform(x)