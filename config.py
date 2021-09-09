class Cfg:
    seed = 0xD153A53

    best_dict = "DeepLabV3_resnet101" + '.pth'
    model = best_dict.split('_')[0]

    in_channels = 1
    output_channels = 2
    batch_size = 4
    backbone = best_dict.split('.')[0].split('_')[1]
    pretrained = True

    root_folder = '/mnt/c/Users/dalma/Desktop/AIIJC/CovidSeg/'

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
