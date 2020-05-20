import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import numpy as np
import imageio

'''
The purpose of this program
--> Due to the medical image characteristic,
it would be better to apply augmentation,
take a look at them and discuss with doctors in regards to relevance of augmentation methods.

(instead of converting into tensor using any favorite framework such as Pytorch and put tensors directly into training)


FYI, https://imgaug.readthedocs.io/en/latest/index.html (fantastic augmentation library, please check them out)

codes
    --> each dictionary (hashmap) key - value (name of augmentation method  - augmentation applied)
    --> you could add more keys into hashmap
return
    each augmenation apply to each img file
    and its output is going to be '1_(key).jpg'

ex)
    original image : '1(patientID).jpg'
    augmented image : '1(patientID)_FlipLR.jpg'

***Note***
currently, the program only has single aug application only, i.e. no mix of two augs (You could probably do it with below codes)
'''

from imgaug import augmenters as iaa
import cv2
from glob import glob

class ApplyAug:
    def __init__(self):
        # self.files = glob(PATH + '*.jpg')
        self.augHash = {'FlipLR' : iaa.Fliplr(1),
                    'FlipUD' : iaa.Flipud(1)
                    }
    '''
    ,'AddRandomValue' : iaa.AddElementwise((-40, 40)),
    'SaltAndPepper' : iaa.SaltAndPepper(0.1, per_channel=True),
    'LaplaceNoise' :  iaa.AdditiveLaplaceNoise(scale=0.2*255, per_channel=True),
    'PoissonNoise' : iaa.AdditivePoissonNoise(40),
    'SimpleMultiplication' :  iaa.MultiplyElementwise((0.5, 1.5)),
    'SigmoidContrast' : iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),
    'LinearContrast' : iaa.LinearContrast((0.4, 1.6)),
    'CLAHE' : iaa.AllChannelsCLAHE(clip_limit=(1, 10))
    '''
    def applyAug(self, PATH, OUTPUT_PATH):
        '''
        params : PATH of images & OUTPUT PATH of augmented images
        returns
        '''
        files = glob(PATH + '*.jpg') # jpg or png or tif etc..
        for file in files:
            filename = file.split('/')[-1].split('.')[0]
            originalImage = cv2.imread(file)
            # apply single augmentation method only
            for key in list(self.augHash.keys()):
                imgList = []
                imgList.append(originalImage)
                seq = iaa.Sequential([
                    self.augHash[key]
                ])
                images_aug = seq.augment_images(imgList)
                cv2.imwrite(OUTPUT_PATH + filename + '_' + key + '.jpg', images_aug[0])

    def applyAugLRUD(self, PATH, OUTPUT_PATH):
        '''
        params : PATH of images & OUTPUT PATH of augmented images
        returns
        '''
        files = glob(PATH + '*.jpg') # jpg or png or tif etc..
        for file in files:
            if 'LR' in file:
                filename = file.split('/')[-1].split('.')[0]
                originalImage = cv2.imread(file)
                # apply single augmentation method only

                imgList = []
                imgList.append(originalImage)
                seq = iaa.Sequential([
                    self.augHash['FlipUD']
                    ])
                images_aug = seq.augment_images(imgList)
                cv2.imwrite(OUTPUT_PATH + filename + 'UD' + '.jpg', images_aug[0])

    def applyAugOnSpecific(self, IMG_PATH, OUTPUT_PATH, oversampleN):
        '''
        --> apply augmentation on specific single data
        params :
        IMG_PATH : single image path
        OUTPUT_PATH : where the file will be saved
        oversampleN : How many aug do you want from augmentation hashmap?

        returns
        list of augmented filename
        '''
        # files = glob(PATH + '*.jpg') # jpg or png or tif etc..
        # for file in files:
        filename = IMG_PATH.split('/')[-1].split('.')[0]
        originalImage = cv2.imread(IMG_PATH)
        augmethods = []
        # apply single augmentation method only
        for key in list(self.augHash.keys())[:oversampleN]:

            imgList = []
            imgList.append(originalImage)
            seq = iaa.Sequential([
                self.augHash[key]
            ])
            images_aug = seq.augment_images(imgList)
            cv2.imwrite(OUTPUT_PATH + filename + '_' + key + '.jpg', images_aug[0])
            augmethods.append(filename + '_' + key)
        return augmethods


def main():
    list = ['0ZERO','1ONE','2TWO','3THREE']
    for i in list:
        PATH = '/home/psj/Desktop/FUNDUS_480_SPLIT_UNDER/train/'+i+'/'
        OUTPUT_PATH = '/home/psj/Desktop/FUNDUS_480_SPLIT_UNDER/train/'+i+'/'
        # print(augHash.keys())
        aug = ApplyAug()
        #print(aug.augHash)
        aug.applyAug(PATH, OUTPUT_PATH)
        aug.applyAugLRUD(OUTPUT_PATH,OUTPUT_PATH)
    return

if __name__ == "__main__":
    main()
