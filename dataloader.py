import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
from glob import glob
%matplotlib inline
data_path = 'datasets/'
tasks = [x for x in os.listdir(data_path) if x.startswith('Task')]

for task in tasks:
    print('Task: ', task)

    # Paths
    path_tr = data_path+task+'/imagesTr'
    path_tr_label = data_path+task+'/labelsTr'
    path_ts = data_path+task+'/imagesTs'
    imglist_tr = glob(path_tr+'/*.gz')
    imglist_tr_label = glob(path_tr_label+'/*.gz')
    imglist_ts = glob(path_ts+'/*.gz')
    print('num_train = {}, num_test = {}'.format(len(imglist_tr), len(imglist_ts)))
    print('Image dimensions:')
    print('Train:')

    # Dimensions
    for img_name in imglist_tr[:5]:
        img = nib.load(img_name)
        print(img.shape)
    print('Test:')
    for img_name in imglist_ts[:5]:
        img = nib.load(img_name)
        print(img.shape)

    # Find number of sub labels

    # Images
    img = nib.load(imglist_tr[0]).get_fdata()
    label = nib.load(imglist_tr_label[0]).get_fdata()
    print('Image Min-Max values: Image={},{} and label={},{}'.format(img.max(), img.min(), label.max(), label.min()))
    print('Number of subclasses = ', int(label.max())+1)
    if task=='Task05_Prostate':
        ax = plt.subplot('131')
        ax.imshow(img[:,:,10,0], cmap='gray')
        ax.set_title('Image channel 1')
        ax = plt.subplot('132')
        ax.imshow(img[:,:,10,1], cmap='gray')
        ax.set_title('Image channel 2')
        ax = plt.subplot('133')
        ax.imshow(label[:,:,10], cmap='gray')
        ax.set_title('Segmentation Mask')
        plt.show()
    else:
        ax = plt.subplot('121')
        ax.imshow(img[:,:,10], cmap='gray')
        ax.set_title('Input image')
        ax = plt.subplot('122')
        ax.imshow(label[:,:,10], cmap='gray')
        ax.set_title('Segmentation Mask')
        plt.show()
        print('\n')
