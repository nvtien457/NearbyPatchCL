import torch
import os
from PIL import Image
import random
import pandas as pd

class ISICDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir:str, transform=None):
        assert os.path.exists(data_dir)
        self.transform = transform

        self.classes = {
            'MEL': 0,
            'NV': 1,
            'BCC': 2,
            'AK': 3,
            'BKL': 4,
            'DF': 5,
            'VASC': 6, 
            'SCC': 7,
            'UNK': 8,
        }

        self.missing_image_path = []
        self.list_image_path = []
        self.gt = []

        df = pd.read_csv(data_dir + '/ISIC_2019_Training_GroundTruth.csv')

        for index, row in df.iterrows():
            p = data_dir + '/ISIC_2019_Training_Input/' + row['image'] + '.jpg'
            if os.path.exists(p):
                self.list_image_path.append(p)
                self.gt.append(row[self.classes.keys()].to_numpy().argmax())
            else:
                self.missing_image_path.append(p)
            
        print('Missing images:', len(self.missing_image_path))

    def __getitem__(self, idx):
        image_path = self.list_image_path[idx]
        label = self.gt[idx]

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label
        

    def __len__(self):
        return len(self.list_image_path)