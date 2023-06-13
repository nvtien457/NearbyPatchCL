import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import ImageFolder
import torch
import torchvision
from torchvision import models, transforms
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
# from nets import *
import time, os, copy, argparse
import multiprocessing
from matplotlib import pyplot as plt
# from model import *
import sklearn
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, ConfusionMatrixDisplay
from torchvision.models import resnet50, resnet18


FOLDER_PATH =  f"D:/DATA/model/SupCon_256/274"
#finetune_e275_p100
FINETUNE_NAME = 'ckpt_274_100/finetune_e30_p100_new_new'
MODEL   = f"D:/DATA/model/SupCon_256/274/ckpt_274.pth"
split = MODEL.split('/')
FOLDER_PATH= MODEL.replace(f'/{split[-1]}','')
# FOLDER_PATH =  f"D:/DATA/model/SupCon_256/149"
# #finetune_e275_p100
# FINETUNE_NAME = 'ckpt_best_149_50/finetune_e30_p50_new'
# MODEL   = f"D:/DATA/model/SupCon_256/149/ckpt_best_149.pth"
# FOLDER_PATH =  f"D:/DATA/model/SimCLR_1024/274"
# #finetune_e275_p100
# FINETUNE_NAME = 'ckpt_274_100/finetune_e30_p100'
# MODEL   = f"D:/DATA/model/SimCLR_1024/274/ckpt_274.pth"
CLASSI0 = f"{FOLDER_PATH}/{FINETUNE_NAME}/fold_0.pth"
CLASSI1 = f"{FOLDER_PATH}/{FINETUNE_NAME}/fold_1.pth"
CLASSI2 = f"{FOLDER_PATH}/{FINETUNE_NAME}/fold_2.pth"
CLASSI3 = f"{FOLDER_PATH}/{FINETUNE_NAME}/fold_3.pth"
CLASSI4 = f"{FOLDER_PATH}/{FINETUNE_NAME}/fold_4.pth"
TEST_PATH= 'E:/medical/Skin_Cancer_Detection_WSI-main/cheat/TEST_SET'

ADD= MODEL.split('/')[-3]+'_'+FINETUNE_NAME.split('/')[0]
def softmax(x):
    e = np.exp(x)
    return e / np.sum(e)
def write_path(list_path, bool):
    filename=''
    if bool ==True:
        filename='positive'
    else:
        filename='negative'
    if(len(list_path)==0):
        return 0
    split = list_path[0].split('\\')
            
    RESULT_PATH = f'E:/medical/Skin_Cancer_Detection_WSI-main/cheat/RESULT/RESULT_{ADD}/{split[-3]}/{split[-2]}'
    isExist = os.path.exists(RESULT_PATH )
    if not isExist:
    # Create a new directory because it does not exist
        os.makedirs(RESULT_PATH )
    with open(f'{RESULT_PATH}/{filename}.txt','w+') as f:
        for path in list_path:
            # split = path.split('\\')
            
            f.write(path)
            f.write('\n')
            
        
    
def main():
# Applying transforms to the data
    image_transforms = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0)),
            transforms.Resize(size=128),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=128),
            # transforms.Resize(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
    }




    # Batch size
    bs = 32
    # Number of workers
    # num_cpu = multiprocessing.cpu_count()
    num_cpu = 2
    # num_cpu = 0

    # Print the train and validation data sizes
    print(MODEL)

    # Set default device as gpu, if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the model for testing
    backbone = 'resnet50'
    backbone = eval(f"{backbone}()")
    backbone.output_dim = backbone.fc.in_features
    backbone.fc = torch.nn.Identity()
    model = backbone
    save_dict = torch.load(MODEL, map_location='cpu')
    model.load_state_dict({k[9:]: v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')},
                                strict=True)
    # model = torch.load(MODEL)
    model.eval()
    model = model.to(device)

    classifier0 = torch.load(CLASSI0)
    classifier0.eval()
    classifier0 = classifier0.to(device)

    classifier1 = torch.load(CLASSI1)
    classifier1.eval()
    classifier1 = classifier1.to(device)

    classifier2 = torch.load(CLASSI2)
    classifier2.eval()
    classifier2 = classifier2.to(device)

    classifier3 = torch.load(CLASSI3)
    classifier3.eval()
    classifier3 = classifier3.to(device)

    classifier4 = torch.load(CLASSI4)
    classifier4.eval()
    classifier4 = classifier4.to(device)


    results = {
        'subset': [],
        'f1': [],
        'balanced_acc': []
    }
    total_pred=[]
    total_true=[]
    for subset in os.listdir(TEST_PATH):
        
        # if subset not in ['PNST']:
        #     continue


        directory = os.path.join(TEST_PATH, subset)
        dataset = ImageFolder(root=directory, transform=image_transforms['valid'])

        print(f'++ {subset}:')

        # Create iterators for data loading
        dataloader = data.DataLoader(dataset, batch_size=bs, shuffle=False,
                                    num_workers=num_cpu, pin_memory=True, drop_last=False)

        # Number of classes
        num_classes = len(dataset.classes)

        print("Validation-set size: {}".format(len(dataset)))


        since = time.time()
        best_acc = 0.0

        model.eval()  # Set model to evaluate mode
        running_corrects = 0
        
        
        #########################
        list_path=dataloader.sampler.data_source.imgs
        categories_list=[]
        for path in list_path:
            category= path[0].split('\\')[-2]
            categories_list.append(category)
        categories_list= list(set(categories_list))
        #'E:/medical/Skin_Cancer_Detection_WSI-main/cheat/TEST_SET\\Histiocytoma\\bg\\Histiocytoma_06_1_101_67.jpg'
        ##########################
        
        positive_list=[]
        negative_list=[]
        
        pred = []
        true = []
        index=0
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            #################################
            # for i in range(bs):
            #     if labels[i] != list_path[index*bs+i][1]:
            #         print(labels[i],list_path[index*32+i][1], index)
                    
            # index += 1
            ##################################
            
            # forward
            # outputs = model(inputs)
            # _, preds = torch.max(outputs, 1)
            # preds = np.zeros(bs)
            feature = model(inputs)
            out_prob0 = np.array(classifier0(feature).detach().cpu())
            out_prob1 = np.array(classifier1(feature).detach().cpu())
            out_prob2 = np.array(classifier2(feature).detach().cpu())
            out_prob3 = np.array(classifier3(feature).detach().cpu())
            out_prob4 = np.array(classifier4(feature).detach().cpu())
            
            preds = np.zeros(len(out_prob0))
            for pred_i in range(len(out_prob0)):
                norm_prob_0 = preprocessing.normalize([out_prob0[pred_i]])
                norm_prob_1 = preprocessing.normalize([out_prob1[pred_i]])
                norm_prob_2 = preprocessing.normalize([out_prob2[pred_i]])
                norm_prob_3 = preprocessing.normalize([out_prob3[pred_i]])
                norm_prob_4 = preprocessing.normalize([out_prob4[pred_i]])
                # norm_prob_0 = softmax(out_prob0[pred_i])
                # norm_prob_1 = softmax(out_prob1[pred_i])
                # norm_prob_2 = softmax(out_prob2[pred_i])
                # norm_prob_3 = softmax(out_prob3[pred_i])
                # norm_prob_4 = softmax(out_prob4[pred_i])
                norm_prob_all = norm_prob_0 + norm_prob_1 + norm_prob_2 + norm_prob_3 + norm_prob_4
                preds[pred_i] = np.argmax(norm_prob_all)

            preds_list = list(preds)
            labels_list = list(np.array(labels.cpu()))
            pred.append(preds_list)
            true.append(labels_list)

        pred = sum(pred, [])
        true = sum(true, [])
        total_pred.append(pred)
        total_true.append(true)
        
        # print(len(pred),len(true))
        for i in range( len(pred)):
            if pred[i]==true[i]:
                positive_list.append(list_path[ i][0])
            else:
                negative_list.append(list_path[ i][0])
                
        for category in categories_list:
            sub_positive_list=[]
            sub_negative_list=[]
            for path in positive_list:
                if category==path.split('\\')[-2]:
                    sub_positive_list.append(path)
            for path in negative_list:
                if category==path.split('\\')[-2]:
                    sub_negative_list.append(path)
            write_path(sub_positive_list,True) 
            write_path(sub_negative_list,False) 
                   

        # epoch_acc = running_corrects.double() / dataset_sizes['valid']
        cm = confusion_matrix(true, pred)
        print(cm)
        f1 = f1_score(true, pred, average='macro')
        print('f1 score:  ', f1)
        # np.savetxt("cm_0221_triple_200.csv", cm, delimiter=",")
        time_elapsed = time.time() - since
        # print('Training complete in {:.0f}m {:.0f}s'.format(
        #     time_elapsed // 60, time_elapsed % 60))
        balance_acc = balanced_accuracy_score(true, pred)
        print('Best balance Acc: {:4f}'.format(balance_acc))

        # print(np.unique(true), np.unique(pred))
        # print(cm)
        # print(dataset.class_to_idx.keys())

        # disp = ConfusionMatrixDisplay.from_predictions(y_true=true, y_pred=pred, labels=list(dataset.class_to_idx.keys()))
        # fig = plt.figure()
        # disp.plot()
        # plt.savefig(f'../checkpoints/{FOLDER_NAME}/{FINETUNE_NAME}/confusion_matrix/{subset}.jpg')

        results['subset'].append(subset)
        results['f1'].append(f1)
        results['balanced_acc'].append(balance_acc)

        # Calculate acc for each class
        one_hot_pred = np.eye(num_classes)[np.array(pred, dtype=int).reshape(-1)]
        one_hot_true = np.eye(num_classes)[np.array(true, dtype=int).reshape(-1)]

        # print(one_hot_pred[:5])
        # print(one_hot_true[:5])
        # print((one_hot_pred * one_hot_true)[:5])
        # print(np.sum(one_hot_pred * one_hot_true, axis=0))
        # print(np.sum(one_hot_true, axis=0))

        compare = np.sum(one_hot_pred * one_hot_true, axis=0) / np.sum(one_hot_true, axis=0)

        for i, c in enumerate(dataset.classes):
            if c not in results.keys():
                results[c] = []
            results[c].append(compare[i])

        print('Class acc:', compare)
    total_true = sum(total_true, [])
    total_pred = sum(total_pred, [])
    cm = confusion_matrix(total_true, total_pred)
    print(cm)
    f1 = f1_score(total_true, total_pred, average='macro')
    print('f1 score:  ', f1)
    # np.savetxt("cm_0221_triple_200.csv", cm, delimiter=",")
    time_elapsed = time.time() - since
    # print('Training complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))
    balance_acc = balanced_accuracy_score(total_true, total_pred)
    print('Best balance Acc: {:4f}'.format(balance_acc))

    # print(np.unique(true), np.unique(pred))
    # print(cm)
    # print(dataset.class_to_idx.keys())
    print(list(dataset.class_to_idx.keys()))
    disp = ConfusionMatrixDisplay.from_predictions(y_true=total_true
                                                   , y_pred=total_pred
                                                   , display_labels=list(dataset.class_to_idx.keys())
                                                   ,normalize='true'
                                                   )
    fig = plt.figure()
    disp.plot()
    plt.savefig(f'{FOLDER_PATH}/{FINETUNE_NAME}/confusion_matrix.jpg')

    results['subset'].append('TOTAL')
    results['f1'].append(f1)
    results['balanced_acc'].append(balance_acc)

    # Calculate acc for each class
    one_hot_pred = np.eye(num_classes)[np.array(total_pred, dtype=int).reshape(-1)]
    one_hot_true = np.eye(num_classes)[np.array(total_true, dtype=int).reshape(-1)]

    # print(one_hot_pred[:5])
    # print(one_hot_true[:5])
    # print((one_hot_pred * one_hot_true)[:5])
    # print(np.sum(one_hot_pred * one_hot_true, axis=0))
    # print(np.sum(one_hot_true, axis=0))

    compare = np.sum(one_hot_pred * one_hot_true, axis=0) / np.sum(one_hot_true, axis=0)

    for i, c in enumerate(dataset.classes):
        if c not in results.keys():
            results[c] = []
        results[c].append(compare[i])

    print('Class acc:', compare)
    df = pd.DataFrame(results)
    print(df)

    # df.to_csv(f'{FOLDER_PATH}/{FINETUNE_NAME}/test_result.csv', index=False)
if __name__ == "__main__":
    main()