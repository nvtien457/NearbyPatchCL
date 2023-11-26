import torch
import os
from PIL import Image
import random
import pandas as pd

class HEDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir:str, transform=None):
        assert os.path.exists(data_dir)
        self.transform = transform
        classes= ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        self.classes = class_to_idx

        self.missing_image_path = []
        self.list_image_path = []
        for category in os.listdir(data_dir):
            for file in os.listdir(os.path.join(data_dir,category)):
                self.list_image_path.append(file)

        
            

    def __getitem__(self, idx):
        image_path = self.list_image_path[idx]

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image
        

    def __len__(self):
        return len(self.list_image_path)