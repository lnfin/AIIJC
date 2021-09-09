class Cfg:
    seed = 0xD153A53

    # DeepLabV3
    model = 'DeepLabV3'

    in_channels = 1
    best_dict = "DeepLabV3_resnet101" + '.pth'
    output_channels = 2
    backbone = 'resnet101'  # 50, 101
    pretrained = True

    # TRAIN AND EVAL SETTINGS
    lr = 1e-4
    epochs = 20
    batch_size = 1

    train_size, val_size = 0.8, 0.2
    # CUSTOM
    metric = 'IoUScore'

    criterion = 'IoULoss'
    optimizer = 'Adam'
    # scheduler = 'OneCycleLR'

    # PATHES
    root_folder = '/mnt/c/Users/dalma/Desktop/AIIJC/CovidSeg/'

    data_folder = 'data/'
    dataset_name = 'MosMed'  # MosMed, Zenodo, ZenodoLungs, MedSeg,

    custom_folder = 'custom/'

    # AUGMENTATIONS AND TRANSFORMS
    pre_transforms = [  # Pre-transforms
        dict(
            name="Resize",
            params=dict(
                height=512,
                width=512,
                p=1.0,
            )
        ),
    ]

    augmentations = [  # Augmentations
        dict(
            name="HorizontalFlip",
            params=dict(
                p=0.5,
            )
        )]

    post_transforms = [  # Post-transforms
        ]

    # CROSS-VALIDATION
    kfold = False

    n_splits = 5
    fold_number = 1  # from 1 to n_splits
