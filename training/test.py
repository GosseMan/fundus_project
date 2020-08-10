from __future__ import print_function, division
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from confusion_matrix import plot_confusion_matrix
from accplot import loss_accuracy_plot
import time
import argparse
import gradcam
import cv2
import roc
import torch.nn.functional as F
import random
rand = 7
torch.manual_seed(rand)
np.random.seed(rand)
random.seed(rand)
'''
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
'''

def test_model(model,image_datasets, dataloaders, criterion, class_num):
    since = time.time()
    dataset_sizes = len(image_datasets)
    print('-' * 10)
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    pred_list = []
    label_list = []
    for i, (inputs, labels) in enumerate(dataloaders,0):
        print(labels[0])
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        pred_list.append(preds[0])
        label_list.append(labels[0])
    epoch_loss = running_loss / dataset_sizes[phase]
    epoch_acc = running_corrects.double() / dataset_sizes[phase]
    print(label_list)
    print(pred_list)
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for i in range(len(label_list)):
        if label_list[i] == 0 and pred_list[i] == 0:
            true_neg += 1
        if label_list[i] == 1 and pred_list[i] == 0:
            false_neg += 1
        if label_list[i] == 0 and pred_list[i] == 1:
            false_pos += 1
        if label_list[i] == 1 and pred_list[i] == 1:
            true_neg += 1
    if class_num == 2:
        sensitivity = true_pos / (true_pos + false_neg)
        specificity = true_neg / (true_neg + false_pos)
        ppv = true_pos / (true_pos + false_pos)
        npv = true_neg / (true_neg + false_neg)
    print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    print('Sensitivity: {:.4f}'.format(sensitivity))
    print('Specificity: {:.4f}'.format(specificity))
    print('PPV: {:.4f}'.format(ppv))
    print('NPV: {:.4f}'.format(npv))
    return True


def roc_curve(model, dataloaders, out_path):
    preds_list = []
    labels_list = []
    model.eval()
    for i, (inputs, labels) in enumerate(dataloaders['val']):
        labels_list = labels_list + labels.tolist()
        inputs = inputs.cuda()
        labels = labels.cuda()
        # 매개변수 경사도를 0으로 설정
        # 순전파
        # 학습 시에만 연산 기록을 추적
        outputs = model(inputs)
        for out in outputs:
            out = F.softmax(out,dim=0).tolist()
            preds_list.append(out[1])
        #_, preds = torch.max(outputs, 1)
        #pred_list = preds.cpu().tolist()
        #preds_list=preds_list+pred_list
    roc.line1(preds_list,labels_list,'./result/'+out_path)

def confusion_mat(model,fig_name, dataloaders,class_names):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    pred=np.array([],dtype='int64')
    ground=np.array([],dtype='int64')
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            ground=np.append(ground,labels.numpy())
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            preds=preds.cpu()
            pred=np.append(pred,preds.numpy())

    plot_confusion_matrix(ground,pred,classes=np.array(class_names),normalize=True)
    while os.path.isfile(fig_name+'_confusion.png'):
        fig_name=fig_name+'-1'
    plt.savefig('./result/'+fig_name+"_confusion.png")


def main():
    parser = argparse.ArgumentParser(description = 'Network')
    parser.add_argument('--gpu_id',type=str,default='0',help='GPU_ID (default:0)')
    parser.add_argument('--lr',type=float,default=0.001,help='Learning rate (default:0)')
    parser.add_argument('--gc',type=bool,default=False,help='GRAD-CAM (default=False)')
    parser.add_argument('--roc',type=bool,default=False,help='ROC-curve (default=False)')
    parser.add_argument('--batch',type=int, default=6,help='Batch_size (default=6)')
    parser.add_argument('--data',type=str, help='Dataset directory name')
    parser.add_argument('--model',type=str, help='Model file name')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    data_dir = '../../data/'+args.data
    image_datasets = datasets.ImageFolder(root = data_dir, data_transforms[x])
    dataloaders = torch.utils.data.DataLoader(image_datasets, 1, shuffle=False, num_workers=4)
    class_names = image_datasets.classes
    print(class_names)

    model_path = "./result/"+args.model
    model_ft = torch.load(model_path, map_location="cuda:0")

    if not os.path.isdir('./result'):
        os.makedirs('./result')

    model_ft = model_ft.cuda()
    criterion = nn.CrossEntropyLoss()
    fig_name = args.model+'_'+args.gpu_id
    (model,image_datasets, dataloaders, criterion)
    test_model(model_ft,image_datasets,dataloaders,criterion, len(class_names))
    #visualize_model(model_ft)
    confusion_mat(model_ft,fig_name,dataloaders,class_names)
    print("time for train : ", time.time()-start)
    if args.gc == True:
        model = model_ft
        target_layer = "features"
        model.eval()
        ########## Case 1: Single file ##########
        data_folder = data_dir
        result_folder = "./result/gradcam_"+args.gpu_id
        while os.path.isdir(result_folder):
            result_folder=result_folder+'-1'
        gcam = gradcam.init_gradcam(model)
        for cls in class_names:
            img_folder = data_folder + '/' + cls
            result_cls_folder = result_folder + '/' + cls
            if not os.path.isdir(result_cls_folder):
                os.makedirs(result_cls_folder)
            for idx, img in enumerate(os.listdir(img_folder)):

                img_path = os.path.join(img_folder, img)

                if os.path.isdir(img_path):
                    continue
                gradcam.single_gradcam(gcam, target_layer, img_path, result_cls_folder, class_names, cls, paper_cmap=True)
    if args.roc == True:
        roc_curve(model_ft,dataloaders, args.gpu_id+'-roc.png')

if __name__ == '__main__':
    main()
