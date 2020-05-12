import pandas as pd
from glob import glob
import os
from shutil import copyfile
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from numpy.random import permutation
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18,resnet34
from torchvision.models.inception import inception_v3
from torchvision.models import densenet121
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
from confusion_matrix import plot_confusion_matrix



is_cuda = torch.cuda.is_available()
is_cuda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def imshow(inp,cmap=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp,cmap)

class FeaturesDataset(Dataset):

    def __init__(self,featlst,labellst):
        self.featlst = featlst
        self.labellst = labellst

    def __getitem__(self,index):
        return (self.featlst[index],self.labellst[index])

    def __len__(self):
        return len(self.labellst)

def fit(epoch,model,data_loader,phase='training',volatile=False):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile=True
    running_loss = 0.0
    running_correct = 0
    for batch_idx , (data,target) in enumerate(data_loader):
        if is_cuda:
            data,target = data.cuda(),target.cuda()
        data , target = Variable(data,volatile),Variable(target)
        if phase == 'training':
            optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output,target)

        running_loss += F.cross_entropy(output,target,size_average=False).data
        preds = output.data.max(dim=1,keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()
    loss = running_loss/len(data_loader.dataset)
    accuracy = 100. * running_correct.item()/len(data_loader.dataset)
    print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')
    if phase == 'validation' and  accuracy> best_acc:
        best_acc = accuracy
        best_model_wts = copy.deepcopy(model.state_dict())
    #model.load_state_dict(best_model_wts)
    return loss,accuracy, model




data_transform = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


train_dset = ImageFolder('./fundus_data_aug/train',transform=data_transform)
val_dset = ImageFolder('./fundus_data_aug/val',transform=data_transform)
classes=4
class_names = train_dset.classes
imshow(train_dset[150][0])

train_loader = DataLoader(train_dset,batch_size=6,shuffle=False,num_workers=1)
val_loader = DataLoader(val_dset,batch_size=6,shuffle=False,num_workers=1)

my_densenet = densenet121(pretrained=True).features
if is_cuda:
    my_densenet = my_densenet.cuda()

for p in my_densenet.parameters():
    p.requires_grad = False

#For training data
trn_labels = []
trn_features = []

#code to store densenet features for train dataset.
for d,la in train_loader:
    o = my_densenet(Variable(d.cuda()))
    o = o.view(o.size(0),-1)
    trn_labels.extend(la)
    trn_features.extend(o.cpu().data)

#For validation data
val_labels = []
val_features = []

#Code to store densenet features for validation dataset.
for d,la in val_loader:
    o = my_densenet(Variable(d.cuda()))
    o = o.view(o.size(0),-1)
    val_labels.extend(la)
    val_features.extend(o.cpu().data)



# Create dataset for train and validation convolution features
trn_feat_dset = FeaturesDataset(trn_features,trn_labels)
val_feat_dset = FeaturesDataset(val_features,val_labels)

# Create data loaders for batching the train and validation datasets
trn_feat_loader = DataLoader(trn_feat_dset,batch_size=6,shuffle=True,drop_last=True)
val_feat_loader = DataLoader(val_feat_dset,batch_size=6)


def confusion_mat(model):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    pred=np.array([],dtype='int64')
    ground=np.array([],dtype='int64')

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_feat_loader):
            ground=np.append(ground,labels.numpy())

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            preds = outputs.data.max(dim=1,keepdim=True)[1]
            preds=preds.cpu()
            pred=np.append(pred,preds.numpy())
    plot_confusion_matrix(ground,pred,classes=np.array(class_names),normalize=True)
    plt.savefig("densenet_confusion_matrix.png")

class FullyConnectedModel(nn.Module):

    def __init__(self,in_size,out_size):
        super().__init__()
        self.fc = nn.Linear(in_size,out_size)

    def forward(self,inp):
        out = self.fc(inp)
        return out



trn_features[0].size(0)

fc_in_size = trn_features[0].size(0)

fc = FullyConnectedModel(fc_in_size,classes)
if is_cuda:
    fc = fc.cuda()


optimizer = optim.Adam(fc.parameters(),lr=0.001)

model_best = densenet121(pretrained=True)
train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]
for epoch in range(1,25):
    epoch_loss, epoch_accuracy , _ = fit(epoch,fc,trn_feat_loader,phase='training')
    val_epoch_loss , val_epoch_accuracy , model_best= fit(epoch,fc,val_feat_loader,phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
confusion_mat(model_best)
