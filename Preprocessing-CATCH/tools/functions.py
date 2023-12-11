import openslide
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import numpy as np
from scipy.ndimage.filters import gaussian_filter

import json
from shapely.geometry import Point, Polygon
import random
from PIL import Image
import os

def extract_svs( filename):
    
    split= filename.split('_')
    svs_name = split[0] + '_' +  split[1] + '_' +  split[2] + '.svs'
    return svs_name
def extract_coord(filename):
    split= filename.split('_')
    x, y = int(split[-2]), int(split[-1].replace('.jpg',''))
    
    return x,y 
def Filter_BG(img, THRESHOLD=0.78):
    convert=np.array(img)

    test_img= convert
    white = np.array([[np.array([255,255,255],dtype=np.uint8)]*512]*512)
    sample = white-test_img

    index = np.argmax(sample,-1)
    max_sample=np.take_along_axis(sample, np.expand_dims(index,-1),-1)
    if THRESHOLD > (max_sample[max_sample<=35].shape)[0]/max_sample[max_sample>=0].shape[0]:
        return True
    return False
def filter_svs_file( img_list):
    svs_files=[]
    for img in img_list:
        split= img.split('_')
        svs_name = split[0] + '_' +  split[1] + '_' +  split[2] + '.svs'
        svs_files.append(svs_name)
    svs_files= list(set(svs_files))
    return svs_files
def generate_image(file, path , DATA_TRAIN_PATH):
    
    print(path)
    path_near= path.replace('TRAIN_SET','TRAIN_SET_NEAR_0')
    file_name = f'{DATA_TRAIN_PATH}/{file}'
        # Open the slide and get the 40x level
        # slide = openslide.open_slide(folder_dir +'\\' + file_name)
    slide = openslide.open_slide(file_name)

    # Set the size of the patches to crop in pixels
    patch_size = 512

    # Calculate the number of patches to crop
    
    num_patches = 1000
    
    # Create a list to store the coordinates and images of the cropped patches
    coords = []

    overlaps_num=0 


    dzg = DeepZoomGenerator(slide, tile_size=patch_size, overlap=overlaps_num, limit_bounds=False)

    level = dzg.level_count -1 # max level

    level_width, level_height = dzg.level_tiles[level]
    mark = np.zeros((level_width, level_height))
    # Generate random coordinates for the patches until the desired number is reached
    count =0
    old_len=len(coords)
    len_count=0
    while len(coords) < num_patches:
        
        if (len_count%100==0 and len_count>0):
            print (len_count)
        
            # Generate random coordinates for the top left corner of the patch
        if old_len== len(coords):
            len_count+=1
        else:
            len_count=0
        if len_count >=2000:
            len_count=0
            break
        x = random.randint(2, level_width -3 )
        y   =   0
        
        arr= np.argwhere(mark[x] == 0)
        arr=arr[arr>=2]
        arr = arr[arr<=level_height -3]
        if len(arr)<=1:
            
            old_len=len(coords)
            continue
        y= arr[random.randint(0, len(arr) -1 )]
        if y==0:
            continue
        if mark[x,y] ==1:
            
            old_len=len(coords)
            continue
        del arr
        
        
        # Get the patch as a NumPy array
        flag=True
        try:
            patch = dzg.get_tile(level, (x, y))
        except:
            flag=False
        # patch = dzg.get_tile(level, (x, y))
        if Filter_BG(patch) == False or flag== False:
            mark[x,y]=1
            mark[x-1,y]=1
            mark[x-1,y-1]=1
            mark[x-1,y+1]=1
            mark[x+1,y]=1
            mark[x+1,y+1]=1
            mark[x+1,y-1]=1
            mark[x,y-1]=1
            mark[x,y+1]=1
            
            old_len=len(coords)
            continue
        
        old_len=len(coords)
        coords.append((x, y))
        count +=1
        
        mark[x,y]=1
        mark[x-1,y]=1
        
        # patch = np.array(patch)
        new_patch = patch.resize((128,128))
        svs_name = file.split('.')[0]
        # x,y =coord
        new_patch.save(f'{path}/{svs_name}_patch_{len(coords)-1:03d}_{x}_{y}.jpg', 'JPEG')
        # coords.append((x, y))
        # count +=1
        # images.append(patch)
        
        del patch
        del new_patch
        # 8 NEARBY PATCHES
        for index in range(0,9):
            if index ==1:
                continue
            index_i= index%3-1
            index_j= (index//3 +1) %3-1
            
            path_near = f'{DATA_PATH}/TRAIN_SET_NEAR_{index}/{tumor}'
            if not os.path.exists(path_near):
                os.makedirs()
            
            file_near= file.replace(f'_{x}_{y}.jpg','')
            if  os.path.exists(f'{path_near}/{file_near}_{index}.jpg'):
                break
            
            patch = dzg.get_tile(level, (x + index_i, y + index_j)) 
            img = patch.resize((128,128))
            img.save(f'{path_near}/{file_near}_{index}.jpg', 'JPEG')
            del img
        image = dzg.get_tile(level, (x - 1, y )) 
        image = image.resize((128,128))
        image.save(f'{path_near}/{svs_name}_patch_{len(coords)-1:03d}_0.jpg', 'JPEG')
        del image

        if count%50 ==0:
            print(len(coords))
        
    del mark   
    del dzg
    del coords
    # del images
    del slide
