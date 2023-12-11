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
