import torch
import numpy as np


def discretize_segmentation_maps(probs, threshold=0.5):
    threshold = torch.from_numpy(np.array(threshold)).to(probs.device)
    return probs > threshold


class IoUScore(object):
    def __init__(self, eps=1e-7):
        self.eps = eps

    def __call__(self, y_pred, y_true):
        if len(y_true.shape) < 4:
            y_true = y_true.unsqueeze(0)
        assert y_pred.shape == y_true.shape, str(y_pred.shape) + ' != ' + str(y_true.shape)
        y_pred = torch.argmax(y_pred, 1).float()
        y_true = torch.argmax(y_true, 1).float()
        intersection = torch.sum(y_true * y_pred >= 1, dim=[1, 2])
        union = torch.sum(y_true, dim=[1, 2]) + torch.sum(y_pred, dim=[1, 2]) - intersection
        iou = ((intersection + self.eps) / (union + self.eps)).mean(dim=0)
        return iou


class IoUScoreBinary(object):
    def __init__(self, eps=1e-7, threshold=0.5):
        self.threshold = threshold
        self.eps = eps

    def __call__(self, y_pred, y_true):
        y_pred = discretize_segmentation_maps(y_pred, self.threshold).float()
        y_true = y_true.float()
        intersection = torch.sum(y_true * y_pred, dim=[1, 2])
        union = torch.sum(y_true, dim=[1, 2]) + torch.sum(y_pred, dim=[1, 2]) - intersection
        ious = ((intersection + self.eps) / (union + self.eps)).mean(dim=0)
        return ious
