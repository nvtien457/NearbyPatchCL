import sys
print(sys.path)

import os
# import openslide
# from openslide import open_slide
# from openslide.deepzoom import DeepZoomGenerator
import random
import numpy as np
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
import json
# import winsound
import torch
import torch.utils.data as data
from datasets import ImageFolder
from shapely.geometry import Point, Polygon
# from fastai.vision import Path, DataLoader
from tqdm import tqdm
from torchvision import models, transforms
from sklearn import preprocessing
from torchvision.models import resnet50
import pickle


def main():

    # DATA_TRAIN_PATH= 'D:/DATA/TRAIN' 
    # f= open('E:\\medical\\CATCH.json')
    # data= json.load(f)
    # f.close()
    # SVS_PATH= '../CATCH'
    # file_name='Histiocytoma_04_1'
    # slide = open_slide(f'{SVS_PATH}/{file_name}.svs')
    # patch_size=512
    # overlap_factor=0
    # dzg = DeepZoomGenerator(slide, tile_size=patch_size, overlap=overlap_factor, limit_bounds=False)
    # level = dzg.level_count-1
    # shape = slide.dimensions
    # # x_indices = np.arange(0,int((shape[0] // (patch_size*overlap_factor)) + 1))* int(patch_size * overlap_factor)
    # # y_indices = np.arange(0,int((shape[1] // (patch_size*overlap_factor)) + 1))* int(patch_size * overlap_factor)
    # # shape
    # # classification_indices = torch.zeros((0,2)).to(learner.data.device)
    # dzg.level_dimensions[level]
    # level_width, level_height = dzg.level_tiles[level]
    # image_transforms = {
    #         'train': transforms.Compose([
    #             # transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0)),
    #             transforms.Resize(size=128),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406],
    #                                 [0.229, 0.224, 0.225])
    #         ]),
    #         'valid': transforms.Compose([
    #             transforms.Resize((128, 128)),
    #             # transforms.Resize(size=224),
    #             transforms.ToTensor(),
    #             transforms.Normalize([0.485, 0.456, 0.406],
    #                                 [0.229, 0.224, 0.225])
    #         ])
    #     }
    # indices = np.indices((level_width,level_height)).reshape(2,-1).T
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # backbone = 'resnet50'

    # backbone = eval(f"{backbone}()")
    # backbone.output_dim = backbone.fc.in_features
    # backbone.fc = torch.nn.Identity()
    # model = backbone
    # save_dict = torch.load('../checkpoint/OSupCon02_20_256/ckpt_184.pth', map_location='cpu')
    # model.load_state_dict({k[9:]: v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')},
    #                             strict=True)
    # model.eval()
    # model = model.to(device)
    # FOLDER_PATH='../checkpoint/OSupCon02_20_256/ckpt_184_50'
    # FINETUNE_NAME='finetune_e11_p50'
    # CLASSI0 = f"{FOLDER_PATH}/{FINETUNE_NAME}/fold_0.pth"
    # CLASSI1 = f"{FOLDER_PATH}/{FINETUNE_NAME}/fold_1.pth"
    # CLASSI2 = f"{FOLDER_PATH}/{FINETUNE_NAME}/fold_2.pth"
    # CLASSI3 = f"{FOLDER_PATH}/{FINETUNE_NAME}/fold_3.pth"
    # CLASSI4 = f"{FOLDER_PATH}/{FINETUNE_NAME}/fold_4.pth"
    # checkpoint0=  torch.load(CLASSI0)
    # classifier0 = checkpoint0['classifier']
    # classifier0.eval()
    # classifier0 = classifier0.to(device)

    # checkpoint1=  torch.load(CLASSI1)
    # classifier1 = checkpoint1['classifier']
    # classifier1.eval()
    # classifier1 = classifier1.to(device)

    # checkpoint2=  torch.load(CLASSI2)
    # classifier2 = checkpoint2['classifier']
    # classifier2.eval()
    # classifier2 = classifier2.to(device)

    # checkpoint3=  torch.load(CLASSI3)
    # classifier3 = checkpoint3['classifier']
    # classifier3.eval()
    # classifier3 = classifier3.to(device)

    # checkpoint4=  torch.load(CLASSI4)
    # classifier4 = checkpoint4['classifier']
    # classifier4.eval()
    # classifier4 = classifier4.to(device)
    
    # with torch.no_grad():
        
    # # indices.shape
    #     pred = []
    #     index_loader = DataLoader(indices, batch_size=32, shuffle=False)
    #     for ind in tqdm(index_loader,desc='Processing %s' % Path(slide._filename).stem):
            
    #         input_batch = torch.stack(
    #             [image_transforms['valid'](
    #                 dzg.get_tile(level, (int(i[0]), int(i[1])))
    #             ) for i in ind]
    #         )
            
                
    #         # input_batch = torch.from_numpy(np.ndarray(
    #         #     [image_transforms['valid'](Image.fromarray(
    #         #         np.array(dzg.get_tile(level, (int(i[0]), int(i[0]))))
    #         #     )) for i in ind]    
    #         # ))
    #         input_batch = input_batch.to(device, non_blocking=True)
    #         feature = model(input_batch)
    #         out_prob0 = np.array(classifier0(feature).detach().cpu())
    #         out_prob1 = np.array(classifier1(feature).detach().cpu())
    #         out_prob2 = np.array(classifier2(feature).detach().cpu())
    #         out_prob3 = np.array(classifier3(feature).detach().cpu())
    #         out_prob4 = np.array(classifier4(feature).detach().cpu())
    #         preds = np.zeros(len(out_prob0))
    #         for pred_i in range(len(out_prob0)):
    #             norm_prob_0 = preprocessing.normalize([out_prob0[pred_i]])
    #             norm_prob_1 = preprocessing.normalize([out_prob1[pred_i]])
    #             norm_prob_2 = preprocessing.normalize([out_prob2[pred_i]])
    #             norm_prob_3 = preprocessing.normalize([out_prob3[pred_i]])
    #             norm_prob_4 = preprocessing.normalize([out_prob4[pred_i]])
                
    #             norm_prob_all = norm_prob_0 + norm_prob_1 + norm_prob_2 + norm_prob_3 + norm_prob_4
    #             preds[pred_i] = np.argmax(norm_prob_all)
    #         preds_list = list(preds)
    #         pred.append(preds_list)
    #     pred = sum(pred, [])    # List of {0...5}
    #     with open('pred.pkl', 'wb') as f:  # open a text file
    #         pickle.dump(pred, f)
    # # pred=[0]*(level_height*level_width)   
    PKL_PATH='../CATCH/SVS' 
    for pkl in os.listdir(PKL_PATH):
        if pkl != 'Melanoma_10_1.svs.pkl':
            continue
        file =open(os.path.join(PKL_PATH,pkl),'rb')
        pred= pickle.load(file)
        colors = {
            0: np.array([255, 255, 255]),
            1: np.array([206, 54, 171]),
            2: np.array([0, 255, 0]),
            3: np.array([128, 0, 128]),
            4: np.array([0, 0, 255]),
            5: np.array([255, 165, 0])
        }
        level_height=170
        level_width=197
        res_img=np.array([])
        print(pred[197*5])

        for i in range(level_width):
            h_img=np.array([])
            for j in range(level_height):
                current_color= colors[pred[j+ level_height*i][0]]
                h,w= pred[j+ level_height*i][1],pred[j+ level_height*i][2]
                h//=4
                w//=4
                img =np.array([[current_color]*w]*h,np.uint8)
                try:
                    if h_img.size==0:
                        h_img = img
                    else:
                        h_img= np.concatenate((h_img,img),axis=0)
                except:
                    print(i,j)
            if res_img.size==0:
                    res_img = h_img
                    
            else:
                res_img= np.concatenate((res_img,h_img),axis=1)
                
            print(i)
        res_img= Image.fromarray(res_img)
        res_img.save(f'{pkl}.png')
        
        
if __name__ == "__main__":
    main()
