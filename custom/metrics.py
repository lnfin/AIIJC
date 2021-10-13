import torch


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
        union = torch.sum(y_true >= 1, dim=[1, 2]) + torch.sum(y_pred >= 1, dim=[1, 2]) - intersection
        iou = ((intersection + self.eps) / (union + self.eps)).mean(dim=0)
        return iou


class DiceScore(object):
    def __init__(self, eps=1e-7):
        self.eps = eps

    def __call__(self, y_pred, y_true):
        if len(y_true.shape) < 4:
            y_true = y_true.unsqueeze(0)
        assert y_pred.shape == y_true.shape, str(y_pred.shape) + ' != ' + str(y_true.shape)
        y_pred = torch.argmax(y_pred, 1).float()
        y_true = torch.argmax(y_true, 1).float()
        intersection = torch.sum(y_true * y_pred >= 1, dim=[1, 2])
        union = torch.sum(y_true >= 1, dim=[1, 2]) + torch.sum(y_pred >= 1, dim=[1, 2])
        dice = ((2 * intersection + self.eps) / (union + self.eps)).mean(dim=0)
        return dice
