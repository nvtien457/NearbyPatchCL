
from tools.functions import *
DATA_PATH = 'D:/DATA/TRAIN/TUMOR' # Folder contents train patches
SVS_PATH ='F:/Data/TRAIN' # Folder contents raw svs for trainning
# 

train_list= os.listdir(SVS_PATH)

for target in train_list:
    tumor = target.split('_')[0]
    
    isExist= os.path.exists(os.path.join(DATA_PATH,'TRAIN_SET',tumor))
    if not isExist:
        os.makedirs(os.path.join(DATA_PATH,'TRAIN_SET',tumor))
        
    for i in range(9):  
        isExist= os.path.exists(os.path.join(DATA_PATH,f'TRAIN_SET_NEAR_{i}',tumor))
        if not isExist:
            os.makedirs(os.path.join(DATA_PATH,f'TRAIN_SET_NEAR_{i}',tumor))
            
    generate_image(target,f'{DATA_PATH}/TRAIN_SET/{tumor}', SVS_PATH)
    
    
    
