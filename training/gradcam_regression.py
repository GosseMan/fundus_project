#!/usr/bin/env python
# coding: utf-8
# In[51]:
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import torch
from torch import nn
from torchvision import models, transforms
from torch.nn import functional as F
# pretrained models import
#import pretrainedmodels
# pretrained EfficientNet import
#from efficientnet_pytorch import EfficientNet
# Opens image from disk, normalizes it and converts to tensor


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)

def get_class_name(c):
    labels = np.loadtxt('./fundus_fazekas.txt', str, delimiter='\t')
    return ' '.join(labels[c].split(',')[0].split()[1:])


def GradCAM(img, c, features_fn, classifier_fn):
    feats = features_fn(img.cuda())
    #print(feats.size())
    #feats = features_fn(img.cuda())
    _, N, H, W = feats.size()
    out = classifier_fn(feats)
    #print(out.size())
    c_score = out[0, c]
    grads = torch.autograd.grad(c_score, feats)
    w = grads[0][0].mean(-1).mean(-1)
    #print(w.size())
    sal = torch.matmul(w, feats.view(N, H*W))
    #print(sal.size())
    sal = F.relu(sal)
    sal = sal.view(H, W).cpu().detach().numpy()
    #print(sal)
    sal = np.maximum(sal, 0)
    #print(sal)
    #print('------------')

    return sal

def cmap_map(function, cmap):
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector
    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

def visualize(img_path, labelfolder,model,outpath):
    read_tensor = transforms.Compose([
        lambda x: Image.open(x),
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
        lambda x: torch.unsqueeze(x, 0)
    ])
    arch = model.__class__.__name__
    #arch = 'DenseNet'
    if arch == 'ResNet':
    #     features_fn = nn.Sequential(*list(model.children())[:-3])
    #     classifier_fn = nn.Sequential(*(list(model.children())[-3:-2] + [Flatten()] + list(model.children())[-2:-1]))
        features_fn = nn.Sequential(*list(model.children())[:-2])
        classifier_fn = nn.Sequential(*(list(model.children())[-2:-1] + [Flatten()] + list(model.children())[-1:]))
    elif arch == 'VGG':
        features_fn = nn.Sequential(*list(model.features.children())[:-1])
        classifier_fn = nn.Sequential(*(list(model.features.children())[-1:] + [Flatten()] + list(model.classifier.children())))
    elif arch == 'DenseNet':
        features_fn = model.features
        classifier_fn = nn.Sequential(*([nn.AdaptiveAvgPool2d(1), Flatten()] + [model.classifier]))
        #classifier_fn = nn.Sequential(*([nn.AvgPool2d(7, 1), Flatten()] + [model.classifier]))
    elif arch == 'InceptionResNetV2':
        #model.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        features_fn = nn.Sequential(*list(model.children())[:-2])
        classifier_fn = nn.Sequential(*(list(model.children())[-2:-1] + [Flatten()] + list(model.children())[-1:]))
    img_tensor = read_tensor(img_path)
    pp, cc = torch.topk(model(img_tensor.cuda()), 1)
    light_jet = cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.jet)
    img_name = os.path.basename(img_path)

    for i, (p, c) in enumerate(zip(pp[0], cc[0])):
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 1, 1)
        sal = GradCAM(img_tensor, int(c), features_fn, classifier_fn)
        img = Image.open(img_path)
        sal = Image.fromarray(sal)
        sal = sal.resize(img.size, resample=Image.LINEAR)
        plt.title('{}: {:.1f}'.format('Predicted Age ', 100*float(p)))
        plt.axis('off')
        plt.imshow(img)
        plt.imshow(np.array(sal), alpha=0.4, cmap=light_jet)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(outpath+'{}/{}_{}_{}'.format(labelfolder, labelfolder, get_class_name(c),img_name),bbox_inces='tight',pad_inches=0,dpi=100)
        plt.close()
    plt.tight_layout()
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1)



def main():
    data_dir = '../'
    #model = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained='imagenet')
    model = torch.load('../3_densenet169_model.pt')
    #model = model_ft
    outpath = '../Seo_gc_conv/'
    if not os.path.isdir(outpath+'features):
        os.makedirs(outpath+'features')
    use_fixed = True
    #model.__class__.__name__
    if use_fixed == True:
        for param in model.parameters():
            param.requires_grad = True
    model = model.eval()
    model = model.cuda()
    #labellist = os.listdir(data_dir+'/val')
    #img_list = ['vk038873-clahe.jpg','vk042499-clahe.jpg','vk080873-clahe.jpg','vk123312-clahe.jpg','vk127891-clahe.jpg']
    img_list = ['vk029159-clahe.jpg','vk029719-clahe.jpg', 'vk029742-clahe.jpg']
    labelfolder = 'features'
    #dirname = data_dir+'/val/{}'.format(labelfolder)
    dirname = data_dir
    filenames = img_list
    #only_name = filename.split('.')[0]
    #count = 0
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    for filename in filenames:
        fullpathname = os.path.join(dirname,filename)
        #img = cv2.imread(fullpathname)
        #size_cropping(img,filename)
        visualize(fullpathname, labelfolder, model, outpath)
    '''
    for labelfolder in labellist:
        dirname = data_dir+'/val/{}'.format(labelfolder)
        filenames = os.listdir(dirname)
        #only_name = filename.split('.')[0]
        #count = 0
        if not os.path.isdir(outpath+labelfolder):
            os.makedirs(outpath+labelfolder)
        for filename in filenames:
            fullpathname = os.path.join(dirname,filename)
            #img = cv2.imread(fullpathname)
            #size_cropping(img,filename)
            visualize(fullpathname, labelfolder, model, outpath)
    '''
if __name__ == '__main__':
    main()
