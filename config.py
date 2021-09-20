class BinaryModelConfig:
    seed = 42
    in_channels = 1
    layers_to_freeze = 0
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
    output_channels = 2

    model = 'UnetPlusPlus'
    backbone = 'resnext101_32x4d'
    encoder_weights = 'swsl'
    best_dict = 'checkpoints/Binary.pth'


class MultiModelConfig:
    seed = 42
    in_channels = 1
    layers_to_freeze = 0
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
    output_channels = 4

    model = 'UnetPlusPlus'
    backbone = 'resnet101'
    best_dict = 'checkpoints/MultiClass.pth'


class LungsModelConfig:
    seed = 42
    in_channels = 1
    layers_to_freeze = 0
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
    output_channels = 2

    model = 'UnetPlusPlus'
    backbone = 'resnet101'
    best_dict = 'checkpoints/Lungs.pth'
