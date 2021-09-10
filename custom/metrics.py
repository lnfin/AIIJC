import torch


class IoUScore(object):
    def __init__(self, eps=1e-7, num_classes=4):
        self.eps = eps
        self.num_classes = num_classes

    def __call__(self, y_pred, y_true):
        y_pred = torch.argmax(torch.sigmoid(y_pred), 1).float()
        y_true = torch.argmax(y_true, 1).float()
        intersection = torch.sum((y_true * y_pred) > 1, dim=[1, 2])
        union = torch.sum(y_true, dim=[1, 2]) + torch.sum(y_pred, dim=[1, 2]) - intersection
        iou = ((intersection + self.eps) / (union + self.eps)).mean(dim=0)
        return iou
