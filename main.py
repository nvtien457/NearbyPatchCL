import torch
import torchvision
from augmentations import get_aug

data_dir = './data'
train = True
transform = get_aug(name='moco', image_size=32, train=True, version='v1')
download = False
distributed = False

dataset = torchvision.datasets.CIFAR10(data_dir, train=train, transform=transform, download=download)
print(len(dataset))

if distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
else:
    train_sampler = None

print(train_sampler)

train_loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=128, 
        shuffle=(train_sampler is None),
        sampler=train_sampler,
    )

for i, ((queries, keys), _) in enumerate(train_loader):
    print(queries.shape, keys.shape)