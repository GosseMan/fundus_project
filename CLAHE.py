import argparse
import cv2
import numpy as np
import torch
import os
from torch.autograd import Function
from torchvision import models

def CLAHE(image_path,gridsize):
    bgr = cv2.imread(image_path)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """
    list = ['0ZERO','1ONE','2TWO']
    for i in list:
        PATH = '/home/psj/Desktop/check/ez/train/'+i+'/'
        OUTPUT_PATH = '/home/psj/Desktop/check/ez/train/'+i+'/'
        filenames = os.listdir(PATH)
        for name in filenames:
            img=CLAHE(PATH+name,7)
            cv2.imwrite(OUTPUT_PATH+name+'-CLAHE.jpg', img)
