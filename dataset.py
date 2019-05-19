from load_function import get_path, get_list

import cv2
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from youtrain.factory import DataFactory
from transforms import test_transform, mix_transform
from tqdm import tqdm
from albumentations.torch import ToTensor
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Sampler
import gc
import json

class BaseDataset(Dataset):
    def __init__(self, path, transform):
        self.ids = get_path(path)
        self.transform = transform
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        raise NotImplementedError


class TrainDataset(BaseDataset):
    def __init__(self, path, transform):
        super().__init__(self, path, transform)

    def __getitem__(self, index):
        name = self.ids[index]
        gl = get_list(name)
        image = gl[0]
        mask = gl[1]
        if 'out_of_focus' not in name:
            mask *= 2
        return self.transform(image=image, mask=mask)

class TestDataset(BaseDataset):
    def __init__(self, image_dir, ids, transform):
        super().__init__(image_dir, ids, transform)
        self.transform = transform
        self.ids = ids
        self.image_dir = image_dir

    def __getitem__(self, index):
        name = self.ids[index]
        image = cv2.imread(os.path.join(self.image_dir, name))
        return self.transform(image=image)['image']


class TaskDataFactory(DataFactory):
    def __init__(self, params, paths, **kwargs):
        super().__init__(params, paths, **kwargs)
        self.fold = kwargs['fold']
        self._folds = None

    @property
    def data_path(self):
        return Path(self.paths['path'])

    def make_transform(self, stage, is_train=False):
        if is_train:
            if stage['augmentation'] == 'mix_transform':
                transform = mix_transform(**self.params['augmentation_params'])
            else:
                raise KeyError('augmentation does not found')
        else:
            transform = test_transform(**self.params['augmentation_params'])
        return transform

    def make_dataset(self, stage, is_train):
        transform = self.make_transform(stage, is_train)
        ids = self.train_ids if is_train else self.val_ids
        return TrainDataset(
            image_dir=self.data_path / self.paths['train_images'],
            mask_dir=self.data_path / self.paths['train_masks'],
            ids=ids,
            transform=transform)

    def make_loader(self, stage, is_train=False):
        dataset = self.make_dataset(stage, is_train)
        return DataLoader(
            dataset=dataset,
            batch_size=self.params['batch_size'],
            shuffle=is_train,
            drop_last=is_train,
            num_workers=self.params['num_workers'],
            pin_memory=torch.cuda.is_available(),
        )
    @property
    def folds(self):
        if self._folds is None:
            self._folds = pd.read_csv(self.data_path / self.paths['folds'])
        return self._folds

    @property
    def train_ids(self):
        return self.folds.loc[self.folds['fold'] != self.fold, 'ImageId'].values

    @property
    def val_ids(self):
        return self.folds.loc[self.folds['fold'] == self.fold, 'ImageId'].values
