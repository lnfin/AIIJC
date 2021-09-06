import numpy as np
import torch


class IoUScore(object):
    def __init__(self, eps=1e-7, threshold=0.5):
        self.threshold = threshold
        self.eps = eps

    def __call__(self, y_pred, y_true):
        y_true = y_true.squeeze()
        print(y_pred.shape, y_true.shape)
        y_pred = torch.argmax(torch.sigmoid(y_pred), -1).float()
        y_true = torch.argmax(y_true, -1).float()
        print(y_pred.shape, y_true.shape)
        intersection = torch.sum(y_true * y_pred, dim=[2, 3])
        union = torch.sum(y_true, dim=[2, 3]) + torch.sum(y_pred, dim=[2, 3]) - intersection
        ious = ((intersection + self.eps) / (union + self.eps)).mean(dim=0)
        return ious
