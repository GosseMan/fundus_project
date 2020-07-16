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
start=time.time()
class MyDensenet169(nn.Module):
    def __init__(self, class_num, pretrained=True):
        super().__init__()
        densenet169 = models.densenet169(pretrained = pretrained)
        num_ftrs = densenet169.classifier.in_features
        densenet169.classifier = nn.Linear(num_ftrs, class_num)
        densenet169 = torch.load('./result/1_densenet169_model.pt')
        #densenet169 = models.densenet169(pretrained=pretrained)
        self.features = nn.ModuleList(densenet169.children())[:-1]
        self.features = nn.Sequential(*self.features)
        # now lets add our new layers
        # num_ftrs = densenet169.classifier.in_features
        self.classifier = nn.Linear(num_ftrs+1, class_num)
    def forward(self, input_imgs, ages):
        # now in forward pass, you have the full control,
        # we can use the feature part from our pretrained model  like this
        out = self.features(input_imgs)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        ages = ages.view(ages.size(0), 1).type(torch.cuda.FloatTensor)
        for i in range(1):
             out = torch.cat((out,ages),dim=1)
        output = self.classifier(out)
        return output, out


def train_model(model,image_datasets, dataloaders,batch_size, criterion, optimizer, scheduler, fig_name, early_stopping, use_meta ,isroc, num_epochs=100):
    since = time.time()
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 10000
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    earlystop = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            '''
            names_list = dataloaders[phase].dataset.samples
            for i in range(len(names_list)):
                names_list[i] = names_list[i][0].split('/')[-1]
            #print(names_list)
            '''
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            for i, (inputs, labels) in enumerate(dataloaders[phase],0):
                inputs = inputs.cuda()
                labels = labels.cuda()
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    if use_meta == False:
                         outputs = model(inputs)
                    else:
                        age_list = dataloaders[phase].dataset.samples[batch_size*i:batch_size*i+len(inputs)]
                        for i in range(len(age_list)):
                            age_list[i] = (float(age_list[i][0].split('/')[-1].split('-')[-1].split('.')[0]))
                        age_list = torch.Tensor(age_list).cuda()
                        outputs, out = model(inputs,age_list)
                        #print(out)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
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
            if phase == 'val':
                if epoch >= 30 and early_stopping==True:
                    if epoch_loss>best_loss:
                        earlystop=earlystop+1
                    else:
                        earlystop = 0
                    if earlystop == 20:
                        print('Early Stopping at Epoch {}'.format(epoch))
                        break
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val Acc at: {} Epoch'.format(best_epoch))
    # 가장 나은 모델 가중치를 불러옴
    fig_name  = './result/'+fig_name
    while os.path.isfile(fig_name+'_lossaccplot.png'):
        fig_name=fig_name+'-1'
    loss_accuracy_plot(num_epochs,train_loss_list,val_loss_list,train_acc_list,val_acc_list,fig_name)
    model.load_state_dict(best_model_wts)
    return model


def roc_curve(model, dataloaders,batch_size, use_meta):
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
        if use_meta == False:
             outputs = model(inputs)
        else:
            age_list = dataloaders['val'].dataset.samples[batch_size*i:batch_size*i+len(inputs)]
            for i in range(len(age_list)):
                age_list[i] = int(age_list[i][0].split('/')[-1].split('-')[-1].split('.')[0])
            age_list = torch.Tensor(age_list).cuda()
            outputs = model(inputs,age_list)
        for out in outputs:
            out = F.softmax(out,dim=0).tolist()
            preds_list.append(out[1])
        #_, preds = torch.max(outputs, 1)
        #pred_list = preds.cpu().tolist()
        #preds_list=preds_list+pred_list
    roc.line1(preds_list,labels_list)


def confusion_mat(model,fig_name, dataloaders,class_names, batch_size, use_meta):
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
            if use_meta == False:
                 outputs = model(inputs)
            else:
                age_list = dataloaders['val'].dataset.samples[batch_size*i:batch_size*i+len(inputs)]
                for i in range(len(age_list)):
                    age_list[i] = int(age_list[i][0].split('/')[-1].split('-')[-1].split('.')[0])
                age_list = torch.Tensor(age_list).cuda()
                outputs, _ = model(inputs,age_list)
            _, preds = torch.max(outputs, 1)
            preds=preds.cpu()
            pred=np.append(pred,preds.numpy())

    plot_confusion_matrix(ground,pred,classes=np.array(class_names),normalize=True)
    while os.path.isfile(fig_name+'_confusion.png'):
        fig_name=fig_name+'-1'
    plt.savefig('./result/'+fig_name+"_confusion.png")


def main():
    parser = argparse.ArgumentParser(description = 'Network')
    parser.add_argument('--network', default='resnet152',type=str)
    parser.add_argument('--gpu_id',type=str,default='0',help='GPU_ID (default:0)')
    parser.add_argument('--lr',type=float,default=0.001,help='Learning rate (default:0)')
    parser.add_argument('--epochs',type=int,default=100,help='Epochs (default:100)')
    parser.add_argument('--fine_tuning',type=bool,default=False,help='Fine Tuning (default=False)')
    parser.add_argument('--class_num',type=int,default=3,help='Class Number (default=3)')
    parser.add_argument('--es',type=bool,default=False,help='Early Stopping (default=False)')
    parser.add_argument('--gc',type=bool,default=False,help='GRAD-CAM (default=False)')
    parser.add_argument('--roc',type=bool,default=False,help='ROC-curve (default=False)')
    parser.add_argument('--batch',type=int, default=6,help='Batch_size (default=6)')
    parser.add_argument('--metadata',type=bool, default=False,help='Use metadata (default=False)')
    parser.add_argument('--data',type=str, help='Dataset directory name')

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
    batch_size = args.batch
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])  for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size,shuffle=True, num_workers=4) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(class_names)
    if args.network == 'resnet18':
        model_ft = models.resnet18(pretrained=True)
        if not args.fine_tuning:
            for param in model_ft.parameters():
                param.requires_grad = False
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, args.class_num)
    elif args.network == 'resnet50':
        model_ft = models.resnet50(pretrained=True)
        if not args.fine_tuning:
            for param in model_ft.parameters():
                param.requires_grad = False
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, args.class_num)
    elif args.network == 'resnet152':
        model_ft = models.resnet152(pretrained=True)
        if not args.fine_tuning:
            for param in model_ft.parameters():
                param.requires_grad = False
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, args.class_num)
    elif args.network == 'densenet121':
        model_ft = models.densenet121(pretrained=True)
        if not args.fine_tuning:
            for param in model_ft.parameters():
                param.requires_grad = False
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, args.class_num)
    elif args.network == 'densenet169':
        if args.metadata == False:
            model_ft = models.densenet169(pretrained=True)
            if not args.fine_tuning:
                for param in model_ft.parameters():
                    param.requires_grad = False
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, args.class_num)
        else:
            model_ft = MyDensenet169(args.class_num)
            if not args.fine_tuning:
                for param in model_ft.parameters():
                    param.requires_grad = False
                for param in model_ft.classifier.parameters():
                    param.requires_grad = True

    else:
        print("Write Network Name")
    if not os.path.isdir('./result'):
        os.makedirs('./result')

    model_ft = model_ft.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(),lr = args.lr)
    steps = int(args.epochs*0.7)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft,milestones=[10,20],gamma=0.1)
    fig_name = args.network+'_'+args.gpu_id
    model_ft=train_model(model_ft,image_datasets,dataloaders,batch_size, criterion,optimizer_ft,
                         exp_lr_scheduler,fig_name, early_stopping=args.es, use_meta = args.metadata,
                         isroc = args.roc,num_epochs=args.epochs)
    #visualize_model(model_ft)
    torch.save(model_ft,'./result/'+args.gpu_id+'_'+args.network+'_model.pt')
    confusion_mat(model_ft,fig_name,dataloaders,class_names, batch_size, use_meta = args.metadata)
    print("time for train : ", time.time()-start)
    if args.gc == True:
        model = model_ft

        if args.network == 'densenet169':
            target_layer = "features"
        else:
            print('Grad-CAM : densenet169 외에는 아직 구현안됨')
        model.eval()

        ########## Case 1: Single file ##########
        data_folder = data_dir+'/val'
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
                result_path = os.path.join(
                    result_cls_folder, img.split(".")[0] + ".jpg"
                    )

                gradcam.single_gradcam(gcam, target_layer, img_path, result_path, class_names, paper_cmap=True)
    if args.roc == True:
        roc_curve(model_ft,dataloaders, batch_size, use_meta = args.metadata)

if __name__ == '__main__':
    main()
