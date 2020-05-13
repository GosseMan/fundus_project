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
start=time.time()


def train_model(model,image_datasets,dataloaders, criterion, optimizer, scheduler,fig_name,num_epochs=100):
    since = time.time()
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 10000
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정
            running_loss = 0.0
            running_corrects = 0
            # 데이터를 반복
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.cuda()
                labels = labels.cuda()
                # 매개변수 경사도를 0으로 설정
                optimizer.zero_grad()
                # 순전파
                # 학습 시에만 연산 기록을 추적
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # 학습 단계인 경우 역전파 + 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # 통계
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train':
                train_loss_list.append(epoch_loss)
                train_acc_list.append(epoch_acc)
            else:
                val_loss_list.append(epoch_loss)
                val_acc_list.append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 모델을 깊은 복사(deep copy)함
            if phase == 'val' and epoch_loss < best_loss:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # 가장 나은 모델 가중치를 불러옴
    loss_accuracy_plot(num_epochs,train_loss_list,val_loss_list,train_acc_list,val_acc_list,fig_name)
    model.load_state_dict(best_model_wts)

    return model


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
    plt.savefig(fig_name+"_confusion.png")

def main():
    parser = argparse.ArgumentParser(description = 'Network')
    parser.add_argument('--network', default='resnet152',type=str)
    parser.add_argument('--gpu_id',type=str,default=0,help='GPU_ID (default:0)')
    parser.add_argument('--lr',type=float,default=0.001,help='Learning rate (default:0)')
    parser.add_argument('--epochs',type=int,default=0,help='Epochs (default:100)')
    parser.add_argument('--fine_tuning',type=bool,default=True,help='Fine Tuning (default=True)')
    parser.add_argument('--class_num',type=int,default=4,help='Class Number (default=4)')
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
    if args.class_num==4:
        data_dir = '../data/FUNDUS_480_SPLIT_UNDER_AUG'
    elif args.class_num==2:
        data_dir = '../data/FUNDUS_480_SPLIT_AUG_BINARY'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])  for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,shuffle=True, num_workers=4) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes


    if args.network == 'resnet18':
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, args.class_num)
    elif args.network == 'resnet152':
        model_ft = models.resnet152(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, args.class_num)
    elif args.network == 'densenet121':
        model_ft = models.densenet121(pretrained=True)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, args.class_num)
    elif args.network == 'densenet169':
        model_ft = models.densenet169(pretrained=True)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, args.class_num)
    else:
        print("Write Network Name")
    if args.fine_tuning==False:
        for param in model_conv.parameters():
            param.requires_grad = False
    model_ft = model_ft.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(),lr = args.lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,step_size=70,gamma=0.1)
    fig_name = args.network+'_'+str(args.class_num)
    model_ft=train_model(model_ft,image_datasets,dataloaders,criterion,optimizer_ft,exp_lr_scheduler,fig_name,num_epochs=args.epochs)
    #visualize_model(model_ft)
    confusion_mat(model_ft,fig_name,dataloaders,class_names)
    print("time : ", time.time()-start)
    return
    
if __name__ == '__main__':
    main()
