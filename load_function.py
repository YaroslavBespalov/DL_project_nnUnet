import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
from glob import glob


def get_path(data_path='datasets/'):
    '''return all paths'''
    tasks = [x for x in os.listdir(data_path) if x.startswith('Task')]
    path = []
    for task in tasks:
        # Paths
        path_tr = data_path + task + '/imagesTr'
        path_tr_label = data_path + task + '/labelsTr'
        imglist_tr = glob(path_tr + '/*.gz')
        imglist_tr_label = glob(path_tr_label + '/*.gz')

        for i in range(len(imglist_tr)):
            path.append([imglist_tr[i], imglist_tr_label[i]])
    return path



def get_list(path):
    '''return all samples in list: [[train_sample],sample_mask]]'''
    res_list = []
    for train_path, label_path in path:
        img = nib.load(train_path).get_fdata()
        img_label = nib.load(label_path).get_fdata()
        if (len(img.shape) != 3):
            for i in range(img.shape[3]):
                res_list.append([img[:, :, :, i] , img_label])
        else:
            res_list.append([img, img_label])
    return res_list
