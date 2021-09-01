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


def datagenerator(cfg):
    image_paths = get_paths(cfg)

    if not cfg.kfold:
        train_paths, val_paths = train_test_split(image_paths, test_size=cfg.test_size, random_state=cfg.seed)
    else:
        kf = KFold(n_splits=cfg.n_splits)
        for i, (train_index, val_index) in enumerate(kf.split(image_paths)):
            if i + 1 == cfg.fold_number:
                train_paths = image_paths[train_index]
                val_paths = image_paths[val_index]

    return train_paths, val_paths


def get_transforms(cfg):
    pretransforms = [getattr(A, item["name"])(**item["params"]) for item in cfg.pretransforms]
    augmentations = [getattr(A, item["name"])(**item["params"]) for item in cfg.augmentations]
    posttransforms = [getattr(A, item["name"])(**item["params"]) for item in cfg.posttransforms]

    print(pretransforms)
    print(augmentations)
    train = A.Compose(pretransforms + augmentations + posttransforms)
    test = A.Compose(pretransforms + posttransforms)
    return train, test


def get_loaders(cfg):
    trainforms, testforms = get_transforms(cfg)

    train_paths, val_paths = datagenerator(cfg)

    train_ds = Covid19Dataset(train_paths, transform=trainforms)
    val_ds = Covid19Dataset(val_paths, transform=trainforms)

    train_dl = DataLoader(train_ds, batch_size=cfg.train_batchsize, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=cfg.val_batchsize, drop_last=False)
    return train_dl, val_dl
