import torch
from torch.utils.data import DataLoader

from datasets import CATCHDataset
from augmentations import TwoCropsTransform, SupConTransform

import time
from tqdm import tqdm
import os

# ds = CATCHDataset(data_dir='../CATCH', name='3', nearby=[0, 7, 2, 4],
#                 transform=TwoCropsTransform(base_transform=SupConTransform(image_size=128)))
# dl = DataLoader(dataset=ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

# t = time.time()
# for i in range(2):
#     for images, labels in tqdm(dl):
#         continue

# print('Elapse time:', time.time() - t)