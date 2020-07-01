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
    if paper_cmap:
        alpha = region[..., None]
        region = alpha * cmap + (1 - alpha) * raw_image
    else:
        region = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(file_path, np.uint8(region))


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


def main():
    '''
    model_type = "Densenet169"
    model_path = "../0630age_result/3_densenet169_model.pt"

    #model = models.load(model_type)
    model.load_state_dict(torch.load(model_path))
    '''
    model = torch.load('./3_densenet169_model.pt')
    model.eval()

    target_layer_lst = ['features.denseblock4', 'features']
    # target_layer = "features.denseblock4.denselayer32"
    #img_list = ['vk038873-clahe.jpg','vk042499-clahe.jpg','vk080873-clahe.jpg','vk123312-clahe.jpg','vk127891-clahe.jpg']
    #img_path = "data/age_resized_clahe_split/val/vk034698.jpg"
    img_list = ['vk029159-clahe.jpg','vk029719-clahe.jpg', 'vk029742-clahe.jpg']
    if not os.path.isdir('./gc'):
        os.makedirs('./gc')
    for img in img_list:
        img_path = './'+img
        print(img_path)

        for target_layer in target_layer_lst:

            result_path = "./gc/"+img.split('.')[0]+'_'+target_layer+'.jpg'



            execute_all(model, target_layer, img_path, result_path, paper_cmap=True)


if __name__ == "__main__":
    main()
