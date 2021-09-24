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
    link = 'https://drive.google.com/uc?id=1uUb8rw8JM6sG9xtaahBrr4SCBcWB70qr'


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
    link = 'https://drive.google.com/uc?id=1W8V3t-TDXH7Bwem6-2I6vYKLmQmOiyXa'


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
    output_channels = 3

    model = 'UnetPlusPlus'
    backbone = 'resnet101'
    best_dict = 'checkpoints/Lungs_L&R.pth'
    link = 'https://drive.google.com/uc?id=1n0evx7Rk0z5MKqo1sXZtX3qkWlwAuTYB'