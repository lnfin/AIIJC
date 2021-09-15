import torch
from config import Cfg
from torch import nn
import sys
from torchvision import models
import segmentation_models_pytorch as smp


class Unet(nn.Module):
    def __init__(self, cfg):
        super(Unet, self).__init__()
        self.cfg = cfg
        self.model = smp.Unet(cfg.backbone, classes=cfg.output_channels, activation='softmax',
                              in_channels=cfg.in_channels)

    def forward(self, x):
        return self.model(x)


class UnetPlusPlus(nn.Module):
    def __init__(self, cfg):
        super(UnetPlusPlus, self).__init__()
        self.cfg = cfg
        self.model = smp.UnetPlusPlus(cfg.backbone, classes=cfg.output_channels, activation='softmax',
                                      in_channels=cfg.in_channels)

    def forward(self, x):
        return self.model(x)


class DeepLabV3(nn.Module):
    def __init__(self, cfg):
        super(DeepLabV3, self).__init__()
        self.cfg = cfg
        self.model = smp.DeepLabV3(cfg.backbone, classes=cfg.output_channels, activation='softmax',
                                   in_channels=cfg.in_channels)

    def forward(self, x):
        return self.model(x)


class OldDeepLabV3(nn.Module):
    base_name = 'deeplabv3_'

    def __init__(self, cfg):
        super(OldDeepLabV3, self).__init__()
        self.cfg = cfg
        self.before_layer = nn.Sequential(nn.Conv2d(self.cfg.in_channels, 3, 1),
                                          nn.BatchNorm2d(3),
                                          nn.ReLU())

        name = self.base_name + self.cfg.backbone
        self.model = getattr(sys.modules['torchvision.models.segmentation'], name)(
            pretrained=self.cfg.pretrained, progress=True
        )

    def forward(self, x):
        if x.shape[1] != 3:
            x = self.before_layer(x)
        return torch.softmax(self.model(x)['out'], dim=1)
