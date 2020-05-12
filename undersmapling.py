import os
import numpy as np
import shutil
import random

# # Creating Train / Val / Test folders (One time use)
root_dir = 'fundus_data'
classes_dir = ['ZERO', 'ONE']
num_sample = 300
#test_ratio = 0.05
#os.makedirs(root_dir+'/test')
for cls in classes_dir:
    os.makedirs(root_dir +'/' + cls + '-under')
    #os.makedirs(root_dir +'/test/' + cls)
    # Creating partitions of the data after shuffeling
    src = root_dir + '/'+cls # Folder to copy images from

    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_FileNames, trash = np.split(np.array(allFileNames),[num_sample])
    train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
    #test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Trash: ', len(trash))
    #print('Testing: ', len(test_FileNames))

    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, root_dir +'/' +cls + '-under')

    #for name in test_FileNames:
        #shutil.copy(name, root_dir +'/test/' + cls)
