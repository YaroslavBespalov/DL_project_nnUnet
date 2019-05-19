import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
from glob import glob

def get_dict(data_path = 'datasets/'):
    tasks = [x for x in os.listdir(data_path) if x.startswith('Task')]

    dict = {}

    for task in tasks:
        # Paths
        path_tr = data_path+task+'/imagesTr'
        path_tr_label = data_path+task+'/labelsTr'
        path_ts = data_path+task+'/imagesTs'
        imglist_tr = glob(path_tr+'/*.gz')
        imglist_tr_label = glob(path_tr_label+'/*.gz')
        imglist_ts = glob(path_ts+'/*.gz')

        value = []
        trains_list = []
        labels_list = []

        for img_name in imglist_tr:
            img = nib.load(img_name).get_fdata()
            if (len(img.shape) != 3):
                for i in range(img.shape[3]):
                    trains_list.append(img[:,:,:,i])
            else:
                    trains_list.append(img)

        modal = int(len(trains_list) / len(imglist_tr))

        for img_name in imglist_tr_label:
            img = nib.load(img_name).get_fdata()
            for i in range(modal):
                labels_list.append(img)

        value.append(trains_list)
        value.append(labels_list)

        dict[task] = value
    return dict
