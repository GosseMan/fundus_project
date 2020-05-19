#!/usr/bin/env python
# coding: utf-8
# In[51]:
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import torch
from torch import nn
from torchvision import models, transforms
# pretrained models import
#import pretrainedmodels




# pretrained EfficientNet import
#from efficientnet_pytorch import EfficientNet


# In[52]:


#model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=5)


# ## Utility functions

# In[53]:


# Opens image from disk, normalizes it and converts to tensor
read_tensor = transforms.Compose([
    lambda x: Image.open(x),
    lambda x: x.convert('RGB'),
    transforms.Resize((480, 480)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
    lambda x: torch.unsqueeze(x, 0)
])
# In[54]:
class Flatten(nn.Module):
    """One layer module that flattens its input."""
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)


# In[55]:

# Given label number returns class name
def get_class_name(c):
    labels = np.loadtxt('./fundus_fazekas.txt', str, delimiter='\t')
    return ' '.join(labels[c].split(',')[0].split()[1:])
# In[56]:
torch.__version__


# ## Load model
# Here, model is split into two parts: feature extractor and classifier. Provided is the implementation for ResNet/VGG/DenseNet architechtures.
#
# Here, `Flatten` layer is being built in in the model as well. In PyTorch implementation flattenning is done in the forward pass, but we need it as a separate layer.

# In[57]:


#model = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained='imagenet')
#model = torch.load('./DataParallel_best.pth')
#model = torch.load('./models/lid_6cls_7_600/DenseNet_best_lid6cls_77_720.pth')
#model = torch.load('./models/lid_6cls_77_750/NASNetALarge_best.pth')
#model = torch.load('./models/mal_7_0508/NASNetALarge_best.pth')
model = torch.load('./model.pt')
#model = torch.load('./models/mal720/DenseNet_best_7_720_0413.pth')
#model = torch.load('./models/5cls720/ResNet_best_5cls_77_720_0414.pth')
use_fixed = True


# In[58]:


#model
#model_ft = models.resnet50(pretrained=True)


# In[59]:


model.__class__.__name__
#torch.save(model, "./models/NAS_best1.pth")
#model.to(device)


# In[60]:


#model = models.vgg19(pretrained=True)
#model = models.resnet50(pretrained=True)
#model = models.densenet201(pretrained=True)
if use_fixed == True:
    for param in model.parameters():
        param.requires_grad = True


# Split model in two parts
arch = model.__class__.__name__
#arch = 'DenseNet'
if arch == 'ResNet':
#     #mal
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


model = model.eval()
model = model.cuda()


# In[61]:


#model.cell_16


# In[62]:


#model


# In[63]:


#classifier_fn


# In[64]:


#features_fn


# ## Grad-CAM

# In[65]:


def GradCAM(img, c, features_fn, classifier_fn):
    feats = features_fn(img.cuda())
    #feats = features_fn(img.cuda())
    _, N, H, W = feats.size()
    out = classifier_fn(feats)
    c_score = out[0, c]
    grads = torch.autograd.grad(c_score, feats)
    w = grads[0][0].mean(-1).mean(-1)
    sal = torch.matmul(w, feats.view(N, H*W))
    sal = sal.view(H, W).cpu().detach().numpy()
    sal = np.maximum(sal, 0)

    return sal


# ## Example

# In[66]:


import matplotlib
import numpy as np
import matplotlib.pyplot as plt

def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
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


# In[67]:


'''
img_path = '../s316-13594 cin1.jpg'
submucosal = []
mucosal = []
'''

def visualize(img_path, labelfolder):
    img_tensor = read_tensor(img_path)
    pp, cc = torch.topk(nn.Softmax(dim=1)(model(img_tensor.cuda())), 1)
    #pp, cc = torch.topk(nn.CrossEntropyLoss()(model(img_tensor.cuda())), 1)
    #pp, cc = torch.topk(outputs, 1)

    #pt, ct = torch.topk(nn.Softmax(dim=1)(model(img_tensor.cuda())), 1)

    #light_jet setting
    light_jet = cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.jet)

    img_name = os.path.basename(img_path)

    print(img_name)


#     plt.figure(figsize=(40, 20))
    for i, (p, c) in enumerate(zip(pp[0], cc[0])):
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 1, i+1)
        sal = GradCAM(img_tensor, int(c), features_fn, classifier_fn)
        img = Image.open(img_path)
        sal = Image.fromarray(sal)
        sal = sal.resize(img.size, resample=Image.LINEAR)

#         height = int(img.height)
#         width = img.width
#         figsize = (1, height/width) if height>=width else (width/height, 1)

        plt.title('{}: {:.1f}%'.format(get_class_name(c), 100*float(p)))
        plt.axis('off')
        plt.imshow(img)
        plt.imshow(np.array(sal), alpha=0.4, cmap=light_jet)


        print(get_class_name(c))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('./{}/{}_{}_{}'.format(labelfolder, labelfolder, get_class_name(c),img_name),bbox_inces='tight',pad_inches=0,dpi=100)


#         if get_class_name(c) == 'Submucosal':
#             submucosal.append(img_name)
#             plt.axis('off')
#             plt.tight_layout()
#             plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1)
#             #plt.savefig('/home/mlm08/ml/data/SUBMUCOSAL/Submucosal_6/gradcam/{}'.format(img_name),bbox_inces='tight',pad_inches=0,dpi=100)
#             plt.savefig('/home/mlm08/ml/data/SUBMUCOSAL/Submucosal_6/gradcam/submucosal/{}'.format(img_name),bbox_inces='tight',pad_inches=0,dpi=100)
#         else:
#             mucosal.append(img_name)
#             plt.axis('off')
#             plt.tight_layout()
#             plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1)
#             #plt.savefig('/home/mlm08/ml/data/SUBMUCOSAL/Submucosal_6/gradcam/{}'.format(img_name),bbox_inces='tight',pad_inches=0,dpi=100)
#             plt.savefig('/home/mlm08/ml/data/SUBMUCOSAL/Submucosal_6/gradcam/mucosal/{}'.format(img_name),bbox_inces='tight',pad_inches=0,dpi=100)


#         plt.contour(np.array(sal), alpha=0.2, cmap='jet')
#         plt.figure(figsize=figsize)
#         plt.axis('off'), plt.xticks([]), plt.yticks([])
#         plt.tight_layout()
#         plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
#         plt.savefig('./test.png',bbox_inces='tight',pad_inches=0,dpi=100)
#         plt.savefig('./grad_images/{}'.format(img_name),bbox_inces='tight',pad_inches=0,dpi=100)

    #plt.show()
    #plt.savefig('/home/mlm08/ml/data/SUBMUCOSAL/Submucosal_6/gradcam/{}'.format(img_name))
    #이미지 공백 없애기
    #plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1)



#visualize(img_path, labelfolder)


# In[68]:


labelfolder = '0ZERO'
dirname = './check/ez/val/{}'.format(labelfolder)
#dirname = "C:\\Users\\USER\\Desktop\\LSIL\\add"
filenames = os.listdir(dirname)
#only_name = filename.split('.')[0]
count = 0

for filename in filenames:
    fullpathname = os.path.join(dirname,filename)
    #img = cv2.imread(fullpathname)
    #size_cropping(img,filename)
    visualize(fullpathname, labelfolder)
    count += 1

#     if count > 100:
#         break
    print(count)


# In[ ]:





# In[ ]:
