from albumentations import *
from albumentations.torch import ToTensor
import numpy as np
import cv2
import random
import numpy as np
# from composition import Compose, OneOf, GrayscaleOrColor
# import functional as F
from imgaug import augmenters as iaa
from scipy.ndimage import label


def pre_transform(resize):
    transforms = []
    transforms.append(Resize(resize, resize))
    return Compose(transforms)

def post_transform():
    return Compose([
        Normalize(
            mean=(0.485),
            std=(0.229)),
        ToTensor()])


def mix_transform(resize):
    return Compose([
        pre_transform(resize=resize),
        #Rotate(limit=10, interpolation=cv2.INTER_LINEAR),
       # IAAAdditiveGaussianNoise(p=0.25),
      #  VerticalFlip(),
        HorizontalFlip(),
      #  RandomGamma(),
    #    RandomRotate90(),
        post_transform()
    ])

def test_transform(resize):
    return Compose([
        pre_transform(resize=resize),
        post_transform()]
    )
