import random

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from utils import get_paths
import albumentations as A


class Covid19Dataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform
        self._len = len(self.paths)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        path = self.paths[index]
        loaded = np.load(path)
        image = loaded['image']
        mask = loaded['mask']
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        image = torch.from_numpy(np.array([image], dtype=np.float))
        image = image.type(torch.FloatTensor)
        mask = torch.from_numpy(np.array([mask], dtype=np.uint8))
        return image, mask


def data_generator(cfg):
    image_paths = get_paths(cfg)
    train_paths, val_paths = [], []

    if not cfg.kfold:
        _train_paths, _val_paths = train_test_split(image_paths, test_size=cfg.val_size, random_state=cfg.seed)
        for paths in _train_paths:
            train_paths.extend(paths)
        for paths in _val_paths:
            val_paths.extend(paths)
        random.shuffle(train_paths)
        random.shuffle(val_paths)
        return train_paths, val_paths

    kf = KFold(n_splits=cfg.n_splits)
    for i, (train_index, val_index) in enumerate(kf.split(image_paths)):
        if i + 1 == cfg.fold_number:
            train_paths = image_paths[train_index]
            val_paths = image_paths[val_index]
    random.shuffle(train_paths)
    random.shuffle(val_paths)
    return train_paths, val_paths


def get_transforms(cfg):
    pre_transforms = [getattr(A, item["name"])(**item["params"]) for item in cfg.pre_transforms]
    augmentations = [getattr(A, item["name"])(**item["params"]) for item in cfg.augmentations]
    post_transforms = [getattr(A, item["name"])(**item["params"]) for item in cfg.post_transforms]

    print(pre_transforms)
    print(augmentations)
    train = A.Compose(pre_transforms + augmentations + post_transforms)
    test = A.Compose(pre_transforms + post_transforms)
    return train, test


def get_loaders(cfg):
    train_transforms, test_transforms = get_transforms(cfg)

    train_paths, val_paths = data_generator(cfg)

    train_ds = Covid19Dataset(train_paths, transform=train_transforms)
    val_ds = Covid19Dataset(val_paths, transform=train_transforms)

    train_dl = DataLoader(train_ds, batch_size=cfg.train_batchsize, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=cfg.val_batchsize, drop_last=False)
    return train_dl, val_dl
