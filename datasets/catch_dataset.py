import torch
import os
from PIL import Image
import random

class CATCHDataset(torch.utils.data.Dataset):
    '''
    CATCH Dataset load from a folder contain .txt files
    Currently use for train self-supervised (not check with supervised, ...)
    Input:
        + data_dir      The directory contains all images (or folder contains images)
        + name          Name of folder dataset (include in /data) contains .txt files saving name of images
        + use_nearby    Return nearby patch (e.g. SimTriplet)
        + patch_id      Nearby patch index  (if use_nearby = True) in range [0, 8]
                        if patch_id = None, random patch_id for each image
        + cancer        list of cancer's names to choose, if None every cancer are chosen
        + transform
    Output:
        List of images
            + use_nearby = True:    (image, nearby)
            + use_nearby = False:   (image)
    '''
    def __init__(self, data_dir:str, name:str, use_nearby:bool = False, patch_id:int = None, cancer:str = None, transform=None):
        assert os.path.exists(data_dir)
        
        available_dataset = os.listdir('data')
        if name not in available_dataset:
            raise ValueError(f'Folder dataset "{name}" not in /data')
        
        if use_nearby and (patch_id not in [None] + [i for i in range(9)]):
            raise ValueError('patch_id out of range [0, 8]')
            
        self.data_dir = data_dir
        self.use_nearby = use_nearby
        self.patch_id = patch_id
        self.transform = transform
        self.folder_dataset_path = os.path.join('data', name)
        
        if cancer is None:
            self.cancer = ['Histiocytoma', 'MCT', 'Melanoma',
                           'Plasmacytoma', 'PNST', 'SCC', 'Trichoblastoma']
        else:
            self.cancer = cancer
        
        self.list_image_path = []
        self.missing_image_path = []
        for file in os.listdir(self.folder_dataset_path):   # .txt file contain path to image
            f = open(os.path.join(self.folder_dataset_path, file), 'r', encoding="utf-8")
            for path_str in f.readlines():
                image_path = os.path.join(self.data_dir, path_str.replace('\n', '') + '.jpg')
                patch_number = int(image_path[-7:-4]) # 000 (..._000.jpg)
                if os.path.exists(image_path):
                    self.list_image_path.append(image_path)
                elif patch_number > 800:
                    self.missing_image_path.append(image_path)
                else:
                    raise ValueError(f'Image {image_path} is not loaded')
            f.close()
            
        print('Missing images:', self.missing_image_path)

    def __getitem__(self, idx):
        image_path = self.list_image_path[idx]

        if self.use_nearby:
            # Use center_image & nearby_image (2 images)
            nearby_image_path = image_path.replace('_SET', '_SET_NEAR')
            nearby_indices = [i for i in range(9)]
            random.shuffle(nearby_indices)

            if self.patch_id is not None:   # e.g. patch_id = 0
                nearby_indices = [self.patch_id] + nearby_indices   # [0, ...]

            for i, nearby_index in enumerate(nearby_indices):
                nearby_image_path = nearby_image_path.replace('.jpg', '_{:01}.jpg'.format(nearby_index))
                if os.path.exists(nearby_image_path):
                    break

                # no image found with patch_id provided
                if i == 0 and self.patch_id is not None:
                    raise ValueError('Can not find patch_id = {:01} with image {}'.format(self.patch_id, image_path))

                if i == (len(nearby_indices) - 1):
                    raise ValueError('Can not find patch_id in range [{:01}, {:01}] with image {}'.format(0, 9, image_path))
            
            # read image
            origin_img = Image.open(image_path).convert('RGB')
            nearby_img = Image.open(nearby_image_path).convert('RGB')

            if self.transform:
                origin_img = self.transform(origin_img)
                nearby_img = self.transform(nearby_img)
            return origin_img, nearby_img, 0
            
        else:
            # Only 1 image
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, 0
        

    def __len__(self):
        return len(self.list_image_path)