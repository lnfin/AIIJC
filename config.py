class Cfg:
    seed = 0xD153A53
    pretrained = True
    in_channels = 1
    batch_size = 4

    pre_transforms = [
        dict(
            name="Resize",
            params=dict(
                height=512,
                width=512,
                p=1.0,
            )
        ),
    ]

    augmentations = []
    post_transforms = []

    def __init__(self, multi=True):
        if not multi:
            self.model = 'DeepLabV3'
            self.backbone = 'resnet101'
            self.best_dict = self.model + '_' + self.backbone + '.pth'
            self.output_channels = 2
        else:
            self.model = 'DeepLabV3'
            self.backbone = 'resnet101'
            self.best_dict = self.model + '_' + self.backbone + '_multi' + '.pth'
            self.output_channels = 4
