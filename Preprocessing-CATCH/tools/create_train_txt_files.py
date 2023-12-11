import os
import random

def extract_svs( filename):
    
    split= filename.split('_')
    svs_name = split[0] + '_' +  split[1] + '_' +  split[2] + '.svs'
    return svs_name
#######
TUMORS= ['Histiocytoma', 'Melanoma', 'MCT','PNST','Plasmacytoma','Trichoblastoma','SCC']
TXT_PATH= './3_10_20_TXT_FILES/data'
isExitst= os.path.exists(TXT_PATH)
if not isExitst:
    os.makedirs(TXT_PATH)

for type in TUMORS:
    PATH=f'E:/medical/Skin_Cancer_Detection_WSI-main/DATA/TRAIN/TUMOR/TRAIN_SET/{type}'
    tmp = f'TRAIN_SET/{type}'
    svs_list=[]
    train_list_path =os.listdir(PATH)
    for file in train_list_path:
        
        svs= extract_svs(file)
        
        svs_list.append(svs)

    svs_list= list(set(svs_list))
    
    # print(file)
    sizes= [3,10,20,'full']
    prev_list=[]
    for size in sizes:
        
        isExitst = os.path.exists(f'{TXT_PATH}/{size}')
        if not isExitst:
            os.makedirs(f'{TXT_PATH}/{size}')
        
        if size== 'full':
            selected_list=svs_list
        else:    
            target_list= list(set(svs_list).difference(set(prev_list)))
            selected_list= random.sample(target_list, k=size - len(prev_list), )
            
            selected_list.sort()
            prev_list.sort()
            
            
            selected_list= list(set(selected_list).union(set(prev_list)))
            
            selected_list.sort()
        # print(selected_list)
        
        
        prev_list= selected_list
        
        with open(f'{TXT_PATH}/{size}/{type}_{size}.txt','w+') as f:
            
            train_list_path= list(set(train_list_path))
            train_list_path.sort()
            
            for file in train_list_path:
            
                svs= extract_svs(file)
                file = file.split('.jpg')[0]
                if svs in selected_list:
                    f.write(f'TRAIN_SET/{type}/{file}')
                    f.write('\n')
                
            
            
            