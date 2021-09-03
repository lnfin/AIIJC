class Cfg:
    seed = 0xD153A53

    # DeepLabV3
    model = 'DeepLabV3'

    deeplab_inchannels = 1
    deeplab_backbone = 50  # 50, 101
    deeplab_pretrained = True

    # TRAIN AND EVAL SETTINGS
    lr = 1e-4
    epochs = 20
    train_batchsize = 10
    val_batchsize = 10

    train_size, val_size = 0.8, 0.2
    # CUSTOM
    metric = 'IoUScore'

    loss_function = 'IoULoss'
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
        dict(
            name="Normalize",
            params=dict(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            )
        )]

    # CROSS-VALIDATION
    kfold = True

    n_splits = 5
    fold_number = 1  # from 1 to n_splits
