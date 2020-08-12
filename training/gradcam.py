"""This module produces a Grad-CAM image.

This module produces a Grad-CAM image of the input image and the trained model. The model should be
a pytorch model type. The input model makes a result from the input image while collecting gradient
values at the same time. Then, the gradients at the target layer are transformed into color map.
Finally, a Grad-CAM image is produced and saved at the designated path.Counting objects: 3, done.


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
    init_gradcam(model)
    load_image(img_path)
    cal_gradcam(model, gcam, image, target_layer)
    save_gradcam(file_path, region, raw_image, paper_cmap(opt))
    single_gradcam(gcam, target_layer, img_path, gcam_path, paper_cmap(opt))

Classes:
    GradCam(model)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.nn import functional as F
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image

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

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(gcam, self.image_shape, mode="bilinear", align_corners=False)
        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)
        return gcam


def init_gradcam(model):
    """Initialize a GradCAM instance for the input model

    Args:
        model (PyTorch model): Trained PyTorch model file

    Returns:
        gcam (GradCAM): GradCAM instance for generating Grad CAM images
    """

    model.eval()

    for param in model.parameters():
        param.requires_grad = True

    gcam = GradCAM(model=model)
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
    raw_image = Image.open(img_path)
    raw_image = raw_image.convert('RGB')
    loader = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = loader(raw_image).unsqueeze(0).to(device)

    return image, raw_image


def cal_gradcam(gcam, image, target_layer):
    """Calculate the gradients and extract information at the target layer

    Args:
        gcam (GradCAM): GradCAM instance for generating Grad CAM images
        image (PyTorch tensor): Pytorch tensor of the transformed image with normalization (ImageNet stat)
        target_layer (str): Name of the target layer of the model (Must have the same layer name in the model)

    Returns:
        region (ndarray): Grad CAM values of the input image at the target layer
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probs, ids = gcam.forward(image)

    ids_ = ids[0, 0].view(1, 1).to(device)
    gcam.backward(ids=ids_)

    regions = gcam.generate(target_layer=target_layer)
    return regions[0, 0], probs[0,0].item(), ids[0,0].item()


def save_gradcam(file_path, region, raw_image, prob, pred, label_list,paper_cmap=False):
    """Save the Grad CAM image

    Args:
        file_path (str): Path that the Grad CAM image will be saved at
        region (ndarray): Grad CAM values of the input image at the target layer
        raw_image (ndarray): Numpy array of the raw input image
        paper_cmap (bool)(opt): Determine whether the color map is drawn throughout or in part of the image
                                False: Throughout / True: Part

    Returns:
    """
    raw_image = np.array(raw_image)[:,:,::-1]
    region = region.cpu().numpy()
    cmap = cm.jet_r(region)[..., :3] * 255.0
    if paper_cmap:
        alpha = region[..., None]
        region = alpha * cmap *0.6 + (1 - alpha*0.6) * raw_image
    else:
        region = (cmap.astype(np.float) + raw_image.astype(np.float))/2
    #cv2.imwrite(file_path, np.uint8(region))

    txt_img = np.zeros((50,region.shape[1],3),np.uint8)
    #txt_img[:]=(255,255,255)

    vcat = cv2.vconcat((txt_img, np.uint8(region)))
    cv2.putText(vcat,'{}: {:.1f}%'.format(label_list[pred]+' class', prob*100),(10,40), 2, 1,(255,255,255), 2, 0)
    cv2.imwrite(file_path, np.uint8(vcat))

def single_gradcam(gcam, target_layer, img_path, gcam_path,label_list, gt, paper_cmap=True):
    """Make a single Grad CAM image at once. Execute load_image, cal_gradcam, and save_gradcam at once
 
    Args:
        gcam (GradCAM): GradCAM instance for generating Grad CAM images
        target_layer (str): Name of the target layer of the model (Must have the same layer name in the model)
        img_path (str): Path of the original image
        gcam_path (str): Path that the Grad CAM image will be saved at
        paper_cmap (bool)(opt): Determine whether the color map is drawn throughout or in part of the image
                                False: Throughout / True: Part

    Returns:
    """
    image, raw_image = load_image(img_path)

    region, prob, pred = cal_gradcam(gcam, image, target_layer)

    result_path = os.path.join(
            gcam_path, gt + '_' + label_list[pred] + '_' + str(round(prob,2)) + '_' + img_path.split('/')[-1]
            )

    save_gradcam(file_path=result_path, region=region, raw_image=raw_image,prob = prob, pred = pred, label_list = label_list, paper_cmap=paper_cmap)


def main():

    model_path = "./0vs1vs23model.pt"
    target_layer = "features"
    label_list = ['0ZERO','1ONE','2TWO']
    model = torch.load(model_path, map_location="cuda:0")


    ########## Case 1: Single file ##########
    data_folder = "./data/smallset"
    result_folder = "./here"
    cls_list = os.listdir(data_folder)
    gcam = init_gradcam(model)
    pred_list = []
    gt_list = []
    for cls in cls_list:
        img_folder = data_folder + '/' + cls
        result_cls_folder = result_folder + '/' + cls
        if not os.path.isdir(result_cls_folder):
            os.makedirs(result_cls_folder)
        for idx, img in enumerate(os.listdir(img_folder)):
            img_path = os.path.join(img_folder, img)
            if os.path.isdir(img_path):
                continue
            result_path = os.path.join(
                result_cls_folder, img.split(".")[0] + "_" + str(idx) + "_" + target_layer + ".jpg"
                )
            #print(result_path)
            single_gradcam(gcam, target_layer, img_path, result_cls_folder, label_list, cls, paper_cmap=True)



    ########## Case 2: Multiple files in a directory ##########

    # import os

    # target_layers = ["layer1", "layer2", "layer3", "layer4"]

    # img_folder = "./(image folder)"
    # result_folder = "./(result folder)"

    # gcam = init_gradcam(model)

    # for idx, img in enumerate(os.listdir(img_folder)):
    #     img_path = os.path.join(img_folder, img)
    #     if os.path.isdir(img_path):
    #         continue

    #     for tidx, target_layer in enumerate(target_layers):
    #         result_path = os.path.join(
    #             result_folder, img.split(".")[0] + "_" + str(tidx) + "_" + target_layer + ".jpg"
    #         )

    #         single_gradcam(gcam, target_layer, img_path, result_path, paper_cmap=True)

    #     print("{} / {} Finished".format(idx, len(os.listdir(img_folder))))


if __name__ == "__main__":
    main()
