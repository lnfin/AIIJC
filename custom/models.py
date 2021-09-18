import torch
from config import Cfg
from torch import nn
import sys
from torchvision import models
import segmentation_models_pytorch as smp
import torch.nn.functional as F


class Unet(nn.Module):
    def __init__(self, cfg):
        super(Unet, self).__init__()
        self.cfg = cfg
        self.model = smp.Unet(cfg.backbone, classes=cfg.output_channels, activation='softmax',
                              in_channels=cfg.in_channels)

        for i, x in enumerate(self.model.encoder.children()):
            if isinstance(x, torch.nn.Sequential):
                if cfg.layers_to_freeze:
                    for param in x.parameters():
                        param.requires_grad = False
                    cfg.layers_to_freeze -= 1

    def forward(self, x):
        return self.model(x)


class UnetPlusPlus(nn.Module):
    def __init__(self, cfg):
        super(UnetPlusPlus, self).__init__()
        self.cfg = cfg
        if 'resnext' not in cfg.backbone:
            self.model = smp.UnetPlusPlus(cfg.backbone, classes=cfg.output_channels, activation='softmax',
                                          in_channels=cfg.in_channels)
        else:
            self.model = smp.UnetPlusPlus(cfg.backbone, classes=cfg.output_channels, activation='softmax',
                                          in_channels=cfg.in_channels, encoder_weights=cfg.encoder_weights)
        for i, x in enumerate(self.model.encoder.children()):
            if isinstance(x, torch.nn.Sequential):
                if cfg.layers_to_freeze:
                    for param in x.parameters():
                        param.requires_grad = False
                    cfg.layers_to_freeze -= 1

    def forward(self, x):
        return self.model(x)


class DeepLabV3(nn.Module):
    def __init__(self, cfg):
        super(DeepLabV3, self).__init__()
        self.cfg = cfg
        self.model = smp.DeepLabV3(cfg.backbone, classes=cfg.output_channels, activation='softmax',
                                   in_channels=cfg.in_channels)

        for i, x in enumerate(self.model.encoder.children()):
            if isinstance(x, torch.nn.Sequential):
                if cfg.layers_to_freeze:
                    for param in x.parameters():
                        param.requires_grad = False
                    cfg.layers_to_freeze -= 1

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


class KiUnet(nn.Module):

    def __init__(self, cfg):
        super(KiUnet, self).__init__()

        self.encoder1 = nn.Conv2d(cfg.in_channels, 16, 3, stride=1, padding=1)
        self.en1_bn = nn.BatchNorm2d(16)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.en2_bn = nn.BatchNorm2d(32)
        self.encoder3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.en3_bn = nn.BatchNorm2d(64)

        self.decoder1 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.de1_bn = nn.BatchNorm2d(32)
        self.decoder2 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.de2_bn = nn.BatchNorm2d(16)
        self.decoder3 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.de3_bn = nn.BatchNorm2d(8)

        self.decoderf1 = nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.def1_bn = nn.BatchNorm2d(32)
        self.decoderf2 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.def2_bn = nn.BatchNorm2d(16)
        self.decoderf3 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.def3_bn = nn.BatchNorm2d(8)

        self.encoderf1 = nn.Conv2d(1, 16, 3, stride=1,
                                   padding=1)
        self.enf1_bn = nn.BatchNorm2d(16)
        self.encoderf2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.enf2_bn = nn.BatchNorm2d(32)
        self.encoderf3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.enf3_bn = nn.BatchNorm2d(64)

        self.intere1_1 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.inte1_1bn = nn.BatchNorm2d(16)
        self.intere2_1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.inte2_1bn = nn.BatchNorm2d(32)
        self.intere3_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.inte3_1bn = nn.BatchNorm2d(64)

        self.intere1_2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.inte1_2bn = nn.BatchNorm2d(16)
        self.intere2_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.inte2_2bn = nn.BatchNorm2d(32)
        self.intere3_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.inte3_2bn = nn.BatchNorm2d(64)

        self.interd1_1 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.intd1_1bn = nn.BatchNorm2d(32)
        self.interd2_1 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.intd2_1bn = nn.BatchNorm2d(16)
        self.interd3_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.intd3_1bn = nn.BatchNorm2d(64)

        self.interd1_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.intd1_2bn = nn.BatchNorm2d(32)
        self.interd2_2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.intd2_2bn = nn.BatchNorm2d(16)
        self.interd3_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.intd3_2bn = nn.BatchNorm2d(64)

        self.final = nn.Conv2d(8, cfg.output_channels, 1, stride=1, padding=0)

        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        out = F.relu(self.en1_bn(F.max_pool2d(self.encoder1(x), 2, 2)))  # U-Net branch
        out1 = F.relu(
            self.enf1_bn(F.interpolate(self.encoderf1(x), scale_factor=(2, 2), mode='bilinear')))  # Ki-Net branch
        tmp = out
        out = torch.add(out, F.interpolate(F.relu(self.inte1_1bn(self.intere1_1(out1))), scale_factor=(0.25, 0.25),
                                           mode='bilinear'))  # CRFB
        out1 = torch.add(out1, F.interpolate(F.relu(self.inte1_2bn(self.intere1_2(tmp))), scale_factor=(4, 4),
                                             mode='bilinear'))  # CRFB

        u1 = out  # skip conn
        o1 = out1  # skip conn

        out = F.relu(self.en2_bn(F.max_pool2d(self.encoder2(out), 2, 2)))
        out1 = F.relu(self.enf2_bn(F.interpolate(self.encoderf2(out1), scale_factor=(2, 2), mode='bilinear')))
        tmp = out
        out = torch.add(out, F.interpolate(F.relu(self.inte2_1bn(self.intere2_1(out1))), scale_factor=(0.0625, 0.0625),
                                           mode='bilinear'))
        out1 = torch.add(out1, F.interpolate(F.relu(self.inte2_2bn(self.intere2_2(tmp))), scale_factor=(16, 16),
                                             mode='bilinear'))

        u2 = out
        o2 = out1

        out = F.relu(self.en3_bn(F.max_pool2d(self.encoder3(out), 2, 2)))
        out1 = F.relu(self.enf3_bn(F.interpolate(self.encoderf3(out1), scale_factor=(2, 2), mode='bilinear')))
        tmp = out
        out = torch.add(out,
                        F.interpolate(F.relu(self.inte3_1bn(self.intere3_1(out1))), scale_factor=(0.015625, 0.015625),
                                      mode='bilinear'))
        out1 = torch.add(out1, F.interpolate(F.relu(self.inte3_2bn(self.intere3_2(tmp))), scale_factor=(64, 64),
                                             mode='bilinear'))

        ### End of encoder block

        ### Start Decoder

        out = F.relu(self.de1_bn(F.interpolate(self.decoder1(out), scale_factor=(2, 2), mode='bilinear')))  # U-NET
        out1 = F.relu(self.def1_bn(F.max_pool2d(self.decoderf1(out1), 2, 2)))  # Ki-NET
        tmp = out
        out = torch.add(out, F.interpolate(F.relu(self.intd1_1bn(self.interd1_1(out1))), scale_factor=(0.0625, 0.0625),
                                           mode='bilinear'))
        out1 = torch.add(out1, F.interpolate(F.relu(self.intd1_2bn(self.interd1_2(tmp))), scale_factor=(16, 16),
                                             mode='bilinear'))

        out = torch.add(out, u2)  # skip conn
        out1 = torch.add(out1, o2)  # skip conn

        out = F.relu(self.de2_bn(F.interpolate(self.decoder2(out), scale_factor=(2, 2), mode='bilinear')))
        out1 = F.relu(self.def2_bn(F.max_pool2d(self.decoderf2(out1), 2, 2)))
        tmp = out
        out = torch.add(out, F.interpolate(F.relu(self.intd2_1bn(self.interd2_1(out1))), scale_factor=(0.25, 0.25),
                                           mode='bilinear'))
        out1 = torch.add(out1, F.interpolate(F.relu(self.intd2_2bn(self.interd2_2(tmp))), scale_factor=(4, 4),
                                             mode='bilinear'))

        out = torch.add(out, u1)
        out1 = torch.add(out1, o1)

        out = F.relu(self.de3_bn(F.interpolate(self.decoder3(out), scale_factor=(2, 2), mode='bilinear')))
        out1 = F.relu(self.def3_bn(F.max_pool2d(self.decoderf3(out1), 2, 2)))

        out = torch.add(out, out1)  # fusion of both branches

        out = F.relu(self.final(out))  # 1*1 conv

        # out = self.soft(out)

        return out
