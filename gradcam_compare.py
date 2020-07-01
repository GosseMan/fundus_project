"""This module produces a Grad-CAM image.

This module produces a Grad-CAM image of the input image and the trained model. The model should be
a pytorch model type. The input model makes a result from the input image while collecting gradient
values at the same time. Then, the gradients at the target layer are transformed into color map.
Finally, a Grad-CAM image is produced and saved at the designated path.

The source codes of this module were written by Kazuto Nakashima, and modified by Hyunseok Oh.
The original version can be found in http://kazuto1011.github.io.
The following is the docstring from the original code.

#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-26


Several python modules should be pre-installed to utilize this module.
Required modules are as follows:
- PyTorch
- Torchvision
- Numpy
- Matplotlib
- OpenCV


Functions:
    load_image(img_path)
    cal_gradcam(model, image, target_layer)
    save_gradcam(file_path, region, raw_image, paper_cmap(opt))
    execute_all(model, target_layer, img_path, gcam_path, paper_cmap(opt))

Classes:
    GradCam(model)
"""

import os
import torch
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
import matplotlib.cm as cm

import cv2
#import models

# Original code
class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.shape[2:]
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)  # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


# Original code
class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)
        print(grads)
        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)

        gcam = F.interpolate(gcam, self.image_shape, mode="bilinear", align_corners=False)
        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam


def load_image(img_path):
    """Load an image and transform into pytorch tensor with normalization

    Args:
        img_path (str): Path of the original image

    Returns:
        image (PyTorch tensor): Pytorch tensor of the transformed image with normalization (ImageNet stat)
        raw_image (ndarray): Numpy array of the raw input image
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_image = cv2.imread(img_path)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    image = image.unsqueeze(0).to(device)
    return image, raw_image


def cal_gradcam(model, image, target_layer):
    """Calculate the gradients and extract information at the target layer

    Args:
        model (PyTorch model): Trained PyTorch model file
        image (PyTorch tensor): Pytorch tensor of the transformed image with normalization (ImageNet stat)
        target_layer (str): Name of the target layer of the model (Must have the same layer name in the model)

    Returns:
        region (ndarray): Grad CAM values of the input image at the target layer
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    for param in model.parameters():
        param.requires_grad = True

    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(image)
    ids_ = ids[0, 0].view(1, 1).to(device)

    gcam.backward(ids=ids_)

    regions = gcam.generate(target_layer=target_layer)
    return regions[0, 0]


def save_gradcam(file_path, region, raw_image, paper_cmap=False):
    """Save the Grad CAM image

    Args:
        file_path (str): Path that the Grad CAM image will be saved at
        region (ndarray): Grad CAM values of the input image at the target layer
        raw_image (ndarray): Numpy array of the raw input image
        paper_cmap (bool)(opt): Determine whether the color map is drawn throughout or in part of the image
                                False: Throughout / True: Part

    Returns:
    """

    region = region.cpu().numpy()
    cmap = cm.jet_r(region)[..., :3] * 255.0

    cv2.imwrite(file_path, np.uint8(cmap))
    if paper_cmap:
        alpha = region[..., None]
        region = alpha * cmap + (1 - alpha) * raw_image
    else:
        region = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    #cv2.imwrite(file_path, np.uint8(region))


def execute_all(model, target_layer, img_path, gcam_path, paper_cmap=True):
    """Execute the whole process at once

    Args:
        model (PyTorch model): Trained PyTorch model file
        target_layer (str): Name of the target layer of the model (Must have the same layer name in the model)
        img_path (str): Path of the original image
        gcam_path (str): Path that the Grad CAM image will be saved at
        paper_cmap (bool)(opt): Determine whether the color map is drawn throughout or in part of the image
                                False: Throughout / True: Part

    Returns:
    """

    image, raw_image = load_image(img_path)
    region = cal_gradcam(model, image, target_layer)
    save_gradcam(file_path=gcam_path, region=region, raw_image=raw_image, paper_cmap=paper_cmap)



######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################
######################################################################################################################



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


def GradCAM2(img, c, features_fn, classifier_fn):
    feats = features_fn(img.cuda())
    #print(feats.size())
    #feats = features_fn(img.cuda())
    _, N, H, W = feats.size()
    print(feats.size())
    out = classifier_fn(feats)
    print(out.size())
    #print(out.size())
    c_score = out[0, c]
    grads = torch.autograd.grad(c_score, feats)
    print(grads)
    w = grads[0][0].mean(-1).mean(-1)
    print(w.size())
    #print(w.size())
    sal = torch.matmul(w, feats.view(N, H*W))

    #print(sal.size())
    sal = sal.view(H, W).cpu().detach().numpy()
    #print(sal)
    sal = np.maximum(sal, 0)
    print(sal.size())

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
        sal = GradCAM2(img_tensor, int(c), features_fn, classifier_fn)
        img = Image.open(img_path)
        sal = Image.fromarray(sal)
        #sal = sal.resize(img.size, resample=Image.LINEAR)
        plt.title('{}: {:.1f}'.format('Predicted Age ', 100*float(p)))
        plt.axis('off')
        #plt.imshow(img)
        plt.imshow(np.array(sal), alpha=0.4, cmap=light_jet)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(outpath+'{}/{}_{}_{}'.format(labelfolder, labelfolder, get_class_name(c),img_name),bbox_inces='tight',pad_inches=0,dpi=100)
        plt.close()
    plt.tight_layout()
    plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1)



def main():
    data_dir = './'
    #model = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained='imagenet')
    model = torch.load('./3_densenet169_model.pt')
    #model = model_ft
    outpath = './gc_compare/'
    if not os.path.isdir(outpath+'seo'):
        os.makedirs(outpath+'seo')
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
    labelfolder = 'seo'
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



    model = torch.load('./3_densenet169_model.pt')
    model.eval()
    target_layer_lst = ['features']
    #target_layer_lst = ['features.denseblock1','features.denseblock2','features.denseblock3']
    #target_layer_lst = ['features.denseblock4.denselayer32.conv1','features.denseblock4.denselayer32.norm2','features.denseblock4.denselayer32.relu2','features.denseblock4.denselayer32.conv2', 'features']
    # target_layer = "features.denseblock4.denselayer32"
    #img_list = ['vk038873-clahe.jpg','vk042499-clahe.jpg','vk080873-clahe.jpg','vk123312-clahe.jpg','vk127891-clahe.jpg']
    if not os.path.isdir('./gc_compare/Oh'):
        os.makedirs('./gc_compare/Oh')
    for img in img_list:
        img_path = './'+img
        print(img_path)

        for target_layer in target_layer_lst:

            result_path = "./gc_compare/Oh/"+img.split('.')[0]+'_'+target_layer+'.jpg'



            execute_all(model, target_layer, img_path, result_path, paper_cmap=False)


if __name__ == "__main__":
    main()
