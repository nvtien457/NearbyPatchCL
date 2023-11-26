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

# from nets import *
import time, os, copy, argparse
import multiprocessing
from matplotlib import pyplot as plt
from models import get_backbone, get_model
import sklearn
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score, ConfusionMatrixDisplay, accuracy_score
from torchvision.models import resnet50, resnet18
import seaborn as sns
import re 
import yaml
from yaml import load, dump
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
from models import Supervised

from models import Supervised

def softmax(x):
    e = np.exp(x)
    return e / np.sum(e)
class Namespace(object):
    def __init__(self, somedict):
        for key, value in somedict.items():
            assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
            else:
                if value == 'None':
                    self.__dict__[key] = None
                else:
                    self.__dict__[key] = value
    
    def __getattr__(self, attribute):

        raise AttributeError(f"Can not find {attribute} in namespace. Please write {attribute} in your config file(xxx.yaml)!")




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config-file', required=True, type=str, help="xxx.yaml")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--debug_subset_size', type=int, default=8)
    parser.add_argument('--download', action='store_true', help="if can't find dataset, download from web")
    parser.add_argument('--data_dir', type=str, default='/content')
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--ckpt_dir', type=str)
    parser.add_argument('--mem_dir', type=str, default='/content/TRAIN_VAL_SET')
    parser.add_argument('--test_dir', type=str, default='/content/VAL_SET')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval_from', type=str, default=None)
    parser.add_argument('--hide_progress', action='store_true')
    parser.add_argument('--mixed_precision', '-mp', action='store_true', help='Mixed precision traing')

    parser.add_argument('--dist-url', default='127.0.0.1', type=str,
                            help='url used to set up distributed training')
    parser.add_argument('--world-size', default=-1, type=int,
                            help='number of nodes for distributed training')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                            help='Use multi-processing distributed training to launch '
                                'N processes per node, which has N GPUs. This is the '
                                'fastest way to use PyTorch for either single node or '
                                'multi node data parallel training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                            help='distributed backend')
    parser.add_argument('--rank', default=-1, type=int,
                            help='node rank for distributed training')

    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        for key, value in Namespace(yaml.load(f, Loader=yaml.FullLoader)).__dict__.items():
            vars(args)[key] = value

    

    return args
            
def main(args):
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

    MODEL_PATH= args.eval.MODEL_PATH
    MODEL_NAME= args.eval.MODEL_NAME
    CHECKPOINT_NUM = args.eval.CHECKPOINT_NUM
    MODEL = f'{MODEL_PATH}/{MODEL_NAME}/ckpt_{CHECKPOINT_NUM}.pth'
    bs = 128
        # Number of workers
        # num_cpu = multiprocessing.cpu_count()
    num_cpu = 4
    # num_cpu = 0

    # Print the train and validation data sizes
    print(MODEL)

    # Set default device as gpu, if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the model for testing
    model = Supervised(get_backbone('resnet50'), num_classes=6)
    ckpt = torch.load(MODEL, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    # model = torch.load(MODEL)
    model.eval()
    model = model.to(device)

    results = {
        'subset': [],
        'f1': [],
        'balanced_acc': []
    }
    total_pred=[]
    total_true=[]

    TEST_PATH= '../CATCH/TEST_SET'
    for subset in os.listdir(TEST_PATH):
        
        # if subset not in ['Trichoblastoma']:
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
            # print(path)
            category= path[0].split('/')[-2]
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

            out_prob = model(inputs)
            # print(out_prob.shape)
            preds = torch.argmax(out_prob, dim=1)
            # print(preds.shape)

            preds_list = list(np.array(preds.detach().cpu()))
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
        # classes = {
        #     'bg': 0,
        #     'Tumor': 1,
        #     'Dermis': 2,
        #     'Subcutis': 3,
        #     'Epidermis': 4,
        #     'Inflamm-Necrosis': 5,
        # }
        classes = dataset.class_to_idx

        for i, c in enumerate(classes):
            if c not in results.keys():
                results[c] = []
            results[c].append(compare[i])

        print('Class acc:', compare)
        del dataloader

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
    # plt.savefig(f'{MODEL_PATH}/{MODEL_NAME}/ckpt_{CHECKPOINT_NUM}/confusion_matrix.jpg')

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

    for i, c in enumerate(classes):
        if c not in results.keys():
            results[c] = []
        results[c].append(compare[i])

    print('Class acc:', compare)
    df = pd.DataFrame(results)
    print(df)

    df.to_csv(f'{MODEL_PATH}/{MODEL_NAME}/test_result.csv', index=False)

if __name__ == "__main__":
   

    args=get_args()
    main(args)
    # print(args.eval.MODEL)
