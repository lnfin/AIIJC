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
    backbone = 'efficientnet-b7'
    best_dict = 'checkpoints/EfficientNet.pth'
    # link = 'https://drive.google.com/uc?id=1uUb8rw8JM6sG9xtaahBrr4SCBcWB70qr'


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
    output_channels = 2

    model = 'UnetPlusPlus'
    backbone = 'efficientnet-b0'
    best_dict = 'checkpoints/EfficientNetMulti.pth'
    # link = 'https://drive.google.com/uc?id=1W8V3t-TDXH7Bwem6-2I6vYKLmQmOiyXa'
