import torch
import os
from PIL import Image
import random

class CATCHDataset(torch.utils.data.Dataset):
    '''
    CATCH Datset
    Input:
        + center_list   list of folders contain center patch
        + cancer        name of specific cancer to choose, if None every cancer are chosen
        + patch_id      in range [0, 8], if None patch_id will be random
        + transform
    Output:
    '''
    def __init__(self, center_list:str, cancer:str = None, patch_id=None, train=True, transform=None):
        self.center_patch_list = center_list
        self.transform = transform
        self.cancer_type = cancer
        self.patch_id = patch_id

        self.image_list = []

        if cancer is not None:                          # e.g: cancer = Melanoma
            print(f'==> CATCH Dataset with cancer = {cancer}')
            for center_dir in center_list:              # ../CATCH/TRAIN_SET
                for f in os.listdir(center_dir):        # Melanoma_33_1_patch_129.jpg
                    if f.startswith(cancer):
                        self.image_list.append((center_dir, f)) # (../CATCH/TRAIN_SET, Melanoma_33_1_patch_129.jpg)
        
        else:                                           # Get all types of cancer
            print(f'==> CATCH Dataset with all types of cancer')
            for center_dir in center_list:              # ../CATCH/LAB_DEPLOY/TRAIN_SET
                for f in os.listdir(center_dir):        # Histiocytoma_21_2_34056.0_42418.0.jpg
                    self.image_list.append((center_dir, f))     # (../CATCH/LAB_DEPLOY/TRAIN_SET, Histiocytoma_21_2_34056.0_42418.0.jpg)

        self.size = len(self.image_list)

    def __getitem__(self, idx):
        if idx < self.size:
            center_dir, img_name = self.image_list[idx]
            center_img_path = os.path.join(center_dir, img_name)

            nearby_dir = center_dir.replace('_SET', '_SET_NEAR')
            nearby_indices = [i for i in range(9)]
            random.shuffle(nearby_indices)

            if self.patch_id is not None:
                nearby_indices = [self.patch_id] + nearby_indices

            for i, nearby_index in enumerate(nearby_indices):
                nearby_img_name = img_name.replace('.jpg', '_{:03}.jpg'.format(nearby_index))
                nearby_img_path = os.path.join(nearby_dir, nearby_img_name)
                if os.path.exists(nearby_img_path):
                    break

                # no image found
                if i == 0 and self.patch_id != None:
                    raise SyntaxError('Can not find patch_id = {:03}'.format(self.patch_id))

                if i == (len(nearby_indices) - 1):
                    raise SyntaxError('Can not find patch_id in range [{:03}, {:03}]'.format(0, 9))

            # read image
            origin_img = Image.open(center_img_path).convert('RGB')
            nearby_img = Image.open(nearby_img_path).convert('RGB')

            if self.transform:
                origin_1, origin_2 = self.transform(origin_img)
                nearby_1, nearby_2 = self.transform(nearby_img)

            return (origin_1, origin_2, nearby_1), 0
        else:
            raise Exception

    def __len__(self):
        return self.size