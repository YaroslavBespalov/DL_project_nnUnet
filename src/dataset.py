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

#from medicaltorch import transforms as MRI_trans

from tqdm import tqdm
from albumentations.torch import ToTensor
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Sampler
import gc
import json

class BaseDataset(Dataset):
    def __init__(self, path, transform):
        self.ids = get_path(path + '/')
        self.transform = transform
#        print(len(self.ids))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        raise NotImplementedError


class TrainDataset(BaseDataset):
    def __init__(self, path, folds, transform):
        super().__init__(path, transform)
        self.folds = folds
        self.ids = [el for el in self.ids if el[0].split('/')[-1] in folds.ImageId.values]
        self.RESAMPLE_SIZE = 64

    def get_rectangular(self, img):
        if not img.sum():
            return 0, 32, 0, 32, 0, 32
        x, y, z = np.where(img)
        x_min = min(x)
        x_max = max(x)
        y_min = min(y)
        y_max = max(y)
        z_min = min(z)
        z_max = max(z)
        return [x_min, x_max, y_min, y_max, z_min, z_max]

    def crop(self, img, crop_boundaries, margin=5):
        x_min, x_max, y_min, y_max, z_min, z_max = crop_boundaries
        return img[max(0, x_min - margin):min(x_max + 1 + margin, img.shape[0]),
                   max(0, y_min - margin):min(y_max + 1 + margin, img.shape[1]),
                   max(0, z_min - margin):min(z_max + 1 + margin, img.shape[2])]

    def resample(self, num_slices):
        return np.linspace(0, num_slices - 1, self.RESAMPLE_SIZE).astype(int)

    def augment_boundaries(self, crop_boundaries, shift=10, p=0.5):
        if np.random.rand() < p:
            shift_size = np.random.randint(-shift, shift)
            crop_boundaries[0] = crop_boundaries[0] + shift_size
            crop_boundaries[1] = crop_boundaries[1] + shift_size

        if np.random.rand() < p:
            shift_size = np.random.randint(-shift, shift)
            crop_boundaries[2] = crop_boundaries[2] + shift_size
            crop_boundaries[3] = crop_boundaries[3] + shift_size
        return crop_boundaries

    def augment_hfplit(self, imgs, mask, p=0.5):
        if np.random.rand() < p:
            imgs = torch.flip(imgs, dims=[0])
            mask = torch.flip(mask, dims=[0])
            return imgs, mask
            #return imgs[::-1], mask[::-1]
        return imgs, mask

    def __getitem__(self, index):
        name = self.folds.iloc[index].ImageId
       # slice = self.folds.iloc[index].ch_number
        # print(self.ids)
        name = [el for el in self.ids if el[0].split('/')[-1] in name][0]

        gl = get_list(name)
        image = gl[0][0]
        mask = gl[0][1]

       # image = image[:, :, slice]
       # mask = mask[:, :, slice]
        pancreats_mask = (mask > 0).astype(int)
        cancer_mask = (mask > 0).astype(int)
        crop_boundaries = self.get_rectangular(pancreats_mask)
        # crop_boundaries = [100, 340, 180, 360, crop_boundaries[4], crop_boundaries[5]]

        #CUSTOM_AUGMENTATION
        crop_boundaries = self.augment_boundaries(crop_boundaries, shift=10, p=0.5)

        image = (self.crop(image, crop_boundaries)) #+ 1024) / 2048 # * 255
        mask = self.crop(cancer_mask, crop_boundaries)
        # print(mask.mean())

        #3d transform slice by slice
        res_image = []
        res_mask = []
        for i in range(image.shape[2]):
            dict_image = (self.transform(image=image[:, :, i], mask=mask[:, :, i]))
            res_image.append(dict_image["image"])
            res_mask.append(dict_image["mask"])
         #########   end 3d transform slice by slice


        resample_ids = self.resample(image.shape[2])


        h, w, _ = image.shape
        res_image = torch.stack(res_image, dim=0)[resample_ids, :, :]
        res_mask = torch.stack(res_mask, dim=0)[resample_ids, :, : ]

        # Horizon inversion
        res_image, res_mask = self.augment_hfplit(res_image, res_mask, p=0.5)

        x, y, z = res_image.shape
        res_image = (res_image.view(1, x, y, z) + 1024) / 1157 * 255
        res_mask = res_mask.view(1, x, y, z)


        return {"image":res_image, "mask":res_mask}
        # print(len(np.unique(image))==1)

        # mask = mask[:,:,slice]
        # ch1 = (mask == 0).astype(int)
        # ch2 = (mask == 1).astype(int)
        # ch3 = (mask == 2).astype(int)
        # mask = np.stack([ch1, ch2, ch3]).transpose(1, 2, 0)
        # print(image.shape, mask.shape)

        #2d return self.transform(image=image.reshape(image.shape[0], image.shape[1], 1),
        #                      mask=mask)

    def __len__(self):
        return len(self.folds)

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
        folds = self.train_ids if is_train else self.val_ids
#        print(self.data_path)
        return TrainDataset(
            path=str(self.data_path),
#            mask_dir=self.data_path / self.paths['train_masks'],
            folds=folds,
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
        return self.folds.loc[self.folds['fold'] != self.fold]

    @property
    def val_ids(self):
        return self.folds.loc[self.folds['fold'] == self.fold]
