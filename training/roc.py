from glob import glob
import os
import pandas as pd
import numpy as np
from shutil import copyfile
import random
from PIL import Image
from tqdm import tqdm
from sklearn import metrics
import torch
import torchvision
from torchvision import datasets, models, transforms
"""
CIN Binary
"""
import matplotlib.pyplot as plt
'''
predicted_value = [0.090994544, 0.1999211311340332, 6.330013275146484e-05, 0.1921595335006714, 0.09237253665924072, 0.0023711323738098145, 0.0009908080101013184, 0.005674600601196289, 0.1746659278869629, 5.3763389587402344e-05, 0.021797358989715576, 0.001896977424621582, 0.06718045473098755, 0.0025205016136169434, 0.0392533540725708, 0.09953969717025757, 0.15761834383010864, 0.0013739466667175293, 0.017223894596099854, 0.0008005499839782715, 0.37571191787719727, 0.6162159144878387, 0.018681108951568604, 0.009186029434204102, 0.13047951459884644, 9.834766387939453e-05, 7.140636444091797e-05, 0.013781964778900146, 0.05577898025512695, 0.0033153891563415527, 0.2212822437286377, 0.0006381869316101074, 0.008741259574890137, 0.32106268405914307, 0.0015873312950134277, 0.8026215434074402, 0.9316457062959671, 1.71661376953125e-05, 2.0503997802734375e-05, 0.006927192211151123, 0.0021771788597106934, 0.05599498748779297, 0.3071363568305969, 1.1920928955078125e-06, 0.006847083568572998, 0.0006075501441955566, 0.00017023086547851562, 0.10978192090988159, 0.01733487844467163, 0.01333075761795044, 0.0005098581314086914, 0.0037160515785217285, 0.014088869094848633, 0.09913212060928345, 0.0007731318473815918, 0.001804351806640625, 0.3987332582473755, 0.0902954, 0.16007943, 0.12100759, 0.29726467, 0.0014603719, 0.4987675, 0.00031659775, 0.007632819, 0.19423606, 0.27938867, 0.0006644815, 0.3810706, 0.7091621, 0.3102857, 0.23801951, 0.9534644, 0.9423744, 0.9768717, 0.5883901, 0.6372151, 0.29982328, 0.6214525]
actual_class_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# print("AUC : " ,metrics.roc_auc_score(actual_class_list, predicted_value))
fpr, tpr, thresholds = metrics.roc_curve(actual_class_list, predicted_value, pos_label=1)
print("-" * 40)
print("AUC : ", metrics.auc(fpr, tpr))

plt.figure()
plt.plot(fpr, tpr, linestyle = '-', color = 'k', label = 'High-risk vs. Low-risk in the CIN system')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize = 18)
plt.ylabel('True Positive Rate', fontsize = 18)
plt.title('Receiver operating characteristic curve', fontsize = 18)
'''
def line1(pred,act,outpath):
    # print("AUC : " ,metrics.roc_auc_score(actual_class_list, predicted_value))
    fpr, tpr, thresholds = metrics.roc_curve(act, pred, pos_label=1)
    print("-" * 40)
    print("AUC : ", metrics.auc(fpr, tpr))

    plt.figure()
    plt.plot(fpr, tpr, linestyle = '-', color = 'k', label = 'High-risk vs. Low-risk in the CIN system')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize = 18)
    plt.ylabel('True Positive Rate', fontsize = 18)
    plt.title('Receiver operating characteristic curve', fontsize = 18)
    #outpath = './result/roc'
    while os.path.isfile(outpath+'.png'):
        outpath=outpath+'-1'
    plt.savefig(outpath+'.png')
'''
actual_class_list = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
 1, 1, 1, 1, 1, 1, 1, 1, 1]


predicted_value = [3.319766e-05, 0.24049103260040283, 0.021650373935699463, 0.00256425142288208, 0.4261886477470398, 0.7220133, 0.9403813, 0.023422653, 0.7100016, 0.707614, 0.00902502, 0.025441281, 0.5547858, 0.48176715, 0.07258894, 0.26325929164886475, 0.01847177743911743, 0.031615614891052246, 0.05854368209838867, 0.1119149923324585, 0.065883614, 0.0019460607, 0.6056695282459259, 0.002793610095977783, 4.559755325317383e-05, 0.0019525885581970215, 0.24851441383361816, 0.4882000684738159, 0.07584142684936523, 0.015369117259979248, 0.21812540292739868, 0.016956090927124023, 0.38801664113998413, 2.481536e-05, 0.43888574838638306, 0.004126906394958496, 0.05746108293533325, 0.23585748672485352, 0.00048232078552246094, 0.015070855617523193, 0.509319394826889, 0.00921773910522461, 0.008412420749664307, 1.6450881958007812e-05, 0.27236467599868774, 0.911368615925312, 0.5808883905410767, 0.025327861309051514, 0.0018552541732788086, 0.10481864213943481, 0.7817852199077606, 0.6927588880062103, 0.0037584900856018066, 0.6819739639759064, 0.017750918865203857, 0.050779521465301514, 0.0024082064628601074, 0.061843156814575195, 0.11523431539535522, 0.395963191986084, 0.9982210259186104, 0.1254878044128418, 0.3947209119796753, 0.6840431392192841, 0.0005373954772949219, 0.9689110461622477, 0.9747941289097071, 0.8169326, 0.5509459, 0.95472455, 0.7269572, 0.45241594, 0.98656833, 0.008607788, 0.9995347, 0.9810134, 0.76138914, 0.9943422, 0.6619358]



print("AUC : " ,metrics.roc_auc_score(actual_class_list, predicted_value))
fpr, tpr, thresholds = metrics.roc_curve(actual_class_list, predicted_value, pos_label=1)
print("-" * 40)
print("AUC : ", metrics.auc(fpr, tpr))

plt.plot(fpr, tpr, linestyle = '--', color = 'k')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
print("THTHTH")
print(thresholds)
print(len(actual_class_list))
print(len(thresholds))

plt.show()
'''
def main():
    model_path = "./result/1_densenet169_model.pt"
    model = torch.load(model_path, map_location="cuda:0")
    data_dir = "../../data/mFS_3years_binary_split8_under_aug_clahe"
    batch_size = 6
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])  for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size,shuffle=True, num_workers=4) for x in ['train', 'val']}

    preds_list = []
    labels_list = []
    model.eval()
    for i, (inputs, labels) in enumerate(dataloaders['val']):
        labels_list = labels_list + labels.tolist()
        inputs = inputs.cuda()
        labels = labels.cuda()
        # 매개변수 경사도를 0으로 설정
        # 순전파
        # 학습 시에만 연산 기록을 추적

        age_list = dataloaders['val'].dataset.samples[batch_size*i:batch_size*i+len(inputs)]
        for i in range(len(age_list)):
            age_list[i] = int(age_list[i][0].split('/')[-1].split('-')[-1].split('.')[0])
        age_list = torch.Tensor(age_list).cuda()
        outputs = model(inputs,age_list)
        for out in outputs:
            out = F.softmax(out,dim=0).tolist()
            preds_list.append(out[1])
        #_, preds = torch.max(outputs, 1)
        #pred_list = preds.cpu().tolist()
        #preds_list=preds_list+pred_list
    fig_name = model_path.split('/')[-1]
    fig_name = fig_name.split('.')[0]+'-roc'
    line1(preds_list,labels_list,fig_name)

if __name__ == '__main__':
    main()
