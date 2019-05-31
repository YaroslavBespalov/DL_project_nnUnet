import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
from glob import glob


def get_path(data_path='datasets/'):
    '''return all paths'''
    tasks = [x for x in os.listdir(data_path) if x.startswith('Task07_Pancreas_npz')]
    path = []
    for task in tasks:
        # Paths
        path_tr = data_path + task + '/imagesTr'
        path_tr_label = data_path + task + '/labelsTr'
        imglist_tr = glob(path_tr + '/*.npz') #.gz
        imglist_tr_label = glob(path_tr_label + '/*.npz') #.gz

        for i in range(len(imglist_tr)):
            path.append([imglist_tr[i], imglist_tr_label[i]])
    return path



def get_list(path):
    '''return all samples in list: [[train_sample],sample_mask]]'''
    res_list = []
    if isinstance (path[0], str):
        train_path, label_path = path
        img = np.load(train_path)['arr_0'] #nib.load(train_path).get_fdata()
        img_label = np.load(label_path)['arr_0'] #nib.load(label_path).get_fdata()
      ### 2D
        # if (len(img.shape) != 3):
        # #    for ch in range(img.shape[2]):
        #     for i in range(img.shape[3]):
        #         res_list.append([img[:, :, :, i] , img_label])
        # else:
        #    for ch in range(img.shape[2]):
        res_list.append([img, img_label])
    else:
        for train_path, label_path in path:
            img = np.load(train_path)['arr_0']  # nib.load(train_path).get_fdata()
            img_label = np.load(label_path)['arr_0']  # nib.load(label_path).get_fdata()
#             if (len(img.shape) != 3):
# #                for ch in range(img.shape[2]):
#                 for i in range(img.shape[3]):
#                     res_list.append([img[:, :, :, i] , img_label])
#             else:
           #     for ch in range(img.shape[2]):
            res_list.append([img, img_label])
    return res_list

#    for i, el in enumerate(res_list):            
#        channel = np.random.randint(el[0].shape[2])
#        res_list[i] = [el[0][:, :, channel], el[1][:, :, channel]]
#    res = []
#    for i, el in enumerate(res_list):     
#        for i in range(el[0].shape[2]):
#            res.append([el[0], el[1], i])
#    return res
#    return res_list
