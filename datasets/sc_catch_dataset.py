import torch
import os
from PIL import Image
import random

class SCCATCHDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir:str, name:str, nearby=[], cancer:str = None, transform=None):
        assert os.path.exists(data_dir)
        
        available_dataset = os.listdir('data')
        if name not in available_dataset:
            raise ValueError(f'Folder dataset "{name}" not in /data')
        
        if nearby is None:
            nearby = [random.choice([0, 2, 3, 4, 5, 6, 7, 8])]
        self.nearby_indices = nearby
            
        self.data_dir = data_dir
        self.transform = transform
        self.folder_dataset_path = os.path.join('data', name)
        
        if cancer is None:
            self.cancer = ['Histiocytoma', 'MCT', 'Melanoma',
                           'Plasmacytoma', 'PNST', 'SCC', 'Trichoblastoma']
        else:
            self.cancer = cancer
        
        self.list_image_path = []
        self.list_nearby_path = []
        self.missing_image_path = []
        for file in os.listdir(self.folder_dataset_path):   # .txt file contain path to image
            f = open(os.path.join(self.folder_dataset_path, file), 'r', encoding="utf-8")
            for path_str in f.readlines():
                image_path = os.path.join(self.data_dir, path_str.replace('\n', '') + '.jpg')
                pos = image_path.find('patch')
                patch_number = int(image_path[pos+6:pos+9]) # 000 (..._000.jpg)
                if os.path.exists(image_path):
                    nearby_image_path = image_path.replace('_SET', '_SET_NEAR')

                    n = []
                    for i, nearby_index in enumerate(self.nearby_indices):
                      nb_path = nearby_image_path.replace('_NEAR', '_NEAR_{}'.format(nearby_index))
                      idx = nb_path.find('patch') + 10
                      nb_path = nb_path[:idx] + f'{nearby_index}' + '.jpg'
                      if not os.path.exists(nb_path):
                          print(image_path)
                          print(nb_path[idx:])
                          print(nb_path)
                          raise ValueError('Can not find patch_id = {:01} with image {}'.format(nearby_index, image_path))

                      if os.path.exists(nb_path):
                          self.list_image_path.append(nb_path)
                          self.list_nearby_path.append(image_path)
                          n.append(nb_path)
                    
                    self.list_image_path.append(image_path)
                    self.list_nearby_path.append(n[0])

                elif patch_number > 800:
                    self.missing_image_path.append(image_path)
                else:
                    raise ValueError(f'Image {image_path} is not loaded')
            f.close()

        if name == '3':
            self.list_image_path = self.list_image_path[:6400]
            
        print('CATCH dataset level:', name)
        print('Number of missing images:', len(self.missing_image_path))

    def __getitem__(self, idx):
        image_path = self.list_image_path[idx]
        nearby_path = self.list_nearby_path[idx]

        origin_image = Image.open(image_path).convert('RGB')
        nearby_image = Image.open(nearby_path).convert('RGB')

        if self.transform:
            origin_image = self.transform(origin_image)
            nearby_image = self.transform(nearby_image)

        return (origin_image, [nearby_image[0]]), 0
        

    def __len__(self):
        return len(self.list_image_path)