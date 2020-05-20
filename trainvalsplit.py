import os
import numpy as np
import shutil
import random

# # Creating Train / Val / Test folders (One time use)
root_dir = 'fundus_data_aug'
classes_dir = ['0ZERO', '1ONE', '2TWO', '3THREE']

val_ratio = 0.15
#test_ratio = 0.05
os.makedirs(root_dir+'/train')
os.makedirs(root_dir+'/val')
#os.makedirs(root_dir+'/test')
for cls in classes_dir:
    os.makedirs(root_dir +'/train/' + cls)
    os.makedirs(root_dir +'/val/' + cls)
    #os.makedirs(root_dir +'/test/' + cls)
    # Creating partitions of the data after shuffeling
    src = root_dir + '/'+cls # Folder to copy images from

    allFileNames = os.listdir(src)
    random.seed(7)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)* (1 - val_ratio))])


    train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
    val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
    #test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    #print('Testing: ', len(test_FileNames))

    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, root_dir +'/train/' +cls)

    for name in val_FileNames:
        shutil.copy(name, root_dir +'/val/' + cls)

    #for name in test_FileNames:
        #shutil.copy(name, root_dir +'/test/' + cls)
