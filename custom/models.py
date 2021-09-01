from config import Cfg
from torch import nn
from torchvision import models


class DeepLabV3(nn.Module):
    def __init__(self, in_channels=1, resnet=101):
        super().__init__()
        self.cfg = Cfg
        self.before_layer = nn.Sequential(nn.Conv2d(in_channels, 3, 1),
                                          nn.BatchNorm2d(3),
                                          nn.ReLU())
        if resnet == 101:
            self.model = models.segmentation.deeplabv3_resnet101(
                pretrained=self.cfg.deeplab_pretrained, progress=True
            )
        elif resnet == 50:
            self.model = models.segmentation.deeplabv3_resnet50(
                pretrained=self.cfg.deeplab_pretrained, progress=True
            )
        self.model.classifier[-1] = nn.Conv2d(256, 1, 1)

    def forward(self, X):
        if X.shape[1] != 3:
            X = self.before_layer(X)
        return torch.sigmoid(self.model(X)['out'])