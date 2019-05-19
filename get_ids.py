import os
from glob import glob

def get_path(data_path='datasets/'):
    tasks = [x for x in os.listdir(data_path) if x.startswith('Task')]

    path = []

    for task in tasks:
        # Paths
        path_tr = data_path + task + '/imagesTr'
        path_tr_label = data_path + task + '/labelsTr'
        imglist_tr = glob(path_tr + '/*.gz')
        imglist_tr_label = glob(path_tr_label + '/*.gz')

        for path_ in imglist_tr:
            path.append([imglist_tr + path_, imglist_tr_label + path_])
    return path
