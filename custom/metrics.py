import numpy as np
import torch


class IoUScore(object):
    def __init__(self, eps=1e-7, threshold=0.5):
        self.threshold = threshold
        self.eps = eps

    def __call__(self, y_pred, y_true):
        y_pred = discretize_segmentation_maps(y_pred, self.threshold).float()
        y_true = y_true.float()
        intersection = torch.sum(y_true * y_pred, dim=[2, 3])
        union = torch.sum(y_true, dim=[2, 3]) + torch.sum(y_pred, dim=[2, 3]) - intersection
        ious = ((intersection + self.eps) / (union + self.eps)).mean(dim=0)
        return ious


def discretize_segmentation_maps(probs, threshold=0.5):
    threshold = torch.from_numpy(np.array(threshold)).to(probs.device)
    return probs > threshold
