import numpy as np

import argparse
from pathlib import Path

import cv2
import pydoc
import torch

from tqdm import tqdm
from dataset import TestDataset
from inference import PytorchInference
from transforms import test_transform
from torch.utils.data import DataLoader
from youtrain.utils import set_global_seeds, get_config, get_last_save
import torchvision.transforms.functional as F

import warnings
warnings.filterwarnings('ignore')

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

class PytorchInference:
    def __init__(self, device, activation='sigmoid'):
        self.device = device
        self.activation = activation

    @staticmethod
    def to_numpy(images):
        return images.data.cpu().numpy()

    def run_one_predict(self, model, images):
        predictions = model(images)
        if self.activation == 'sigmoid':
            predictions = F.sigmoid(predictions)
        elif self.activation == 'softmax':
            predictions = predictions.exp()
        return predictions

    def predict(self, model, loader):
        model = model.to(self.device).eval()

        with torch.no_grad():
            for data in loader:
                images = data.to(self.device)
                predictions = model(images)
                for prediction in predictions:
                    prediction = np.moveaxis(self.to_numpy(prediction), 0, -1)
                    yield prediction

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--paths', type=str, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    config = get_config(args.config)
    paths = get_config(args.paths)
    params = config['train_params']
    model_name = config['train_params']['model']
    model = pydoc.locate(model_name)(**params['model_params'])
    model.load_state_dict(torch.load(params['weights'])['state_dict'])
    paths = paths['data']

    dataset = TestDataset(
            image_dir=Path(paths['path']) / Path(paths['test_images']),
            ids=None,
            transform=test_transform(**config['data_params']['augmentation_params']))

    loader = DataLoader(
            dataset=dataset,
            batch_size=16,
            shuffle=False,
            drop_last=False,
            num_workers=16,
            pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inferencer = PytorchInference(device)

    for pred in tqdm(inferencer.predict(model, loader), total=len(dataset)):
        pass


if __name__== '__main__':
    main()
