import torch
from config import Cfg
from torch import nn
from torchvision import models


class DeepLabV3(nn.Module):
    def __init__(self, in_channels=1, output_channels=2, resnet=101):
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
        self.model.classifier[-1] = nn.Conv2d(256, output_channels, 1)

    def forward(self, X):
        if X.shape[1] != 3:
            X = self.before_layer(X)
        return torch.sigmoid(self.model(X)['out'])


class UNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=2):
        super().__init__()
        self.cfg = Cfg
        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.pool0 = nn.MaxPool2d(2, stride=2)  # 256 -> 128

        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.pool3 = nn.MaxPool2d(2, stride=2)

        # bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(512, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        # decoder (upsampling)
        self.upsample0 = nn.ConvTranspose2d(1024, 512, 2, stride=2)  # 16 -> 32
        self.dec_conv0 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.upsample1 = nn.ConvTranspose2d(512, 256, 2, stride=2)  # 32 -> 64
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.upsample2 = nn.ConvTranspose2d(256, 128, 2, stride=2)  # 64 -> 128
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.upsample3 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # 128 -> 256
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, output_channels, 1)
        )

    def forward(self, x):
        # encoder
        e0 = self.enc_conv0(x)
        e01 = self.pool0(e0)
        e1 = self.enc_conv1(e01)
        e11 = self.pool1(e1)
        e2 = self.enc_conv2(e11)
        e21 = self.pool2(e2)
        e3 = self.enc_conv3(e21)
        e31 = self.pool3(e3)

        # bottleneck
        b = self.bottleneck_conv(e31)

        # decoder
        d0 = self.dec_conv0(torch.cat((self.upsample0(b), e3), dim=1))
        d1 = self.dec_conv1(torch.cat((self.upsample1(d0), e2), dim=1))
        d2 = self.dec_conv2(torch.cat((self.upsample2(d1), e1), dim=1))
        d3 = self.dec_conv3(torch.cat((self.upsample3(d2), e0), dim=1))  # no activation
        return torch.sigmoid(d3)
