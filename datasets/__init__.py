import torch
import torchvision

from .catch_dataset import CATCHDataset
from .finetune_dataset import FinetuneDataset
from .folder_dataset import ImageFolder

def get_dataset(dataset_cfg, transform=None, debug_subset_size=None):
    dataset = dataset_cfg.name

    if dataset == 'mnist':
        dataset = torchvision.datasets.MNIST(transform=transform, **dataset_cfg.params)
    elif dataset == 'stl10':
        dataset = torchvision.datasets.STL10(transform=transform, **dataset_cfg.params)
    elif dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(transform=transform, **dataset_cfg.params)
    elif dataset == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(transform=transform, **dataset_cfg.params)
    elif dataset == 'imagenet':
        dataset = torchvision.datasets.ImageNet(transform=transform, **dataset_cfg.params)
    
    elif dataset == 'CATCH':
        dataset = CATCHDataset(transform=transform, **dataset_cfg.params)

    else:
        raise NotImplementedError

    if debug_subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, range(0, debug_subset_size)) # take only one batch
        # dataset.classes = dataset.dataset.classes
        # dataset.targets = dataset.dataset.targets
        
    return dataset