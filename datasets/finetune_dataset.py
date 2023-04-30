import torch
import os
from PIL import Image
import random

class FinetuneDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir:str, name:str, transform=None):
        assert os.path.exists(data_dir)
        
        available_dataset = os.listdir('./data/Percentage_finetune')
        if name not in available_dataset:
            raise ValueError(f'Folder dataset "{name}" not exist')
            
        self.data_dir = data_dir
        self.transform = transform
        self.folder_dataset_path = os.path.join('data/Percentage_finetune', name)
        
        self.list_image_path = []
        self.labels = []
        self.missing_image_path = []

        self.classes = {cls: i for i, cls in enumerate(os.listdir(data_dir + '/VAL_SET'))}

        for file in os.listdir(self.folder_dataset_path):   # .txt file contain path to image
            c = file.split('_')[0]

            f = open(os.path.join(self.folder_dataset_path, file), 'r', encoding="utf-8")
            for path_str in f.readlines():
                image_path = os.path.join(self.data_dir, path_str.replace('\n', ''))
                if os.path.exists(image_path):
                    self.list_image_path.append(image_path)
                    self.labels.append(self.classes[c])
                else:
                    self.missing_image_path.append(image_path)
            f.close()
            
        print('Missing images:', self.missing_image_path)

    def __getitem__(self, idx):
        image_path = self.list_image_path[idx]

        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
            
        return image, label
        

    def __len__(self):
        return len(self.list_image_path)