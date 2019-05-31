import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from  albumentations import *
import torch
import pandas as pd
from scipy.ndimage import label


def apk(actual, predicted, k=10):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

class map3(nn.Module):
    def __init__(self, ):
        super(map3, self).__init__()
        
    def forward(self, preds, targs):
        # targs = np.where(targs==1)[1]
        predicted_idxs = preds.sort(descending=True)[1]
        top_3 = predicted_idxs[:, :3]
        res = mapk([[t] for t in targs.cpu().numpy()], top_3.cpu().numpy(), 3)
        return -torch.tensor(res)


class Dice(nn.Module):
    def __init__(self, ):
        super(Dice, self).__init__()
    
    def forward(self, prediction, target):
        smooth = 1e-15
        prediction = prediction.sigmoid()
        prediction = (prediction.view(-1)).float()
        target = target.view(-1)
        dice = (2*torch.sum(prediction * target, dim=0) + smooth) / \
                            (torch.sum(prediction, dim=0) + torch.sum(target, dim=0) + smooth)
        return dice.mean()


class LossMultiLabelDice(nn.Module):
    def __init__(self, dice_weight=1):
        super(LossMultiLabelDice, self).__init__()
        self.dice_weight = dice_weight
        self.smooth = 1e-50

    def dice_coef(self, y_true, y_pred):
        y_true_f = torch.flatten(y_true)
        y_pred_f = torch.flatten(y_pred)
        intersection = torch.sum(y_true_f * y_pred_f)
        return 1 - (2. * intersection + self.smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + self.smooth)

    def dice_coef_multilabel(self, y_true, y_pred, numLabels=3):
        dice = 0

        for index in range(1, numLabels):
            dice += self.dice_coef(y_true[:, index], y_pred[:, index])
        return dice / 2

    def forward(self, outputs, targets):

       # print(outputs.size())
        targets = targets.squeeze().permute(0, 3, 1, 2)
       # print(targets.size())
        loss = 0
        if self.dice_weight:
            loss += self.dice_weight * self.dice_coef_multilabel(outputs, targets)
        return loss
