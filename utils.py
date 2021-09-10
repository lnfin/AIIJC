import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt
import custom.models


def set_seed(seed=0xD153A53):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class OneHotEncoder:
    def __init__(self, cfg):
        self.zeros = [0] * cfg.num_classes

    def encode_num(self, x):
        zeros = self.zeros.copy()
        zeros[int(x)] = 1
        return zeros

    def __call__(self, y):
        y = np.array(y)
        y = np.expand_dims(y, -1)
        y = np.apply_along_axis(self.encode_num, -1, y)
        y = np.swapaxes(y, -1, 1)
        y = np.ascontiguousarray(y)
        return torch.Tensor(y)


def get_metric(cfg):
    return getattr(sys.modules['custom.metrics'], cfg.metric)


def get_criterion(cfg):
    return getattr(sys.modules['custom.losses'], cfg.criterion)


def get_model(cfg):
    name = cfg.model
    model = getattr(sys.modules['custom.models'], name)
    return model


def get_optimizer(cfg):
    optimizer = getattr(sys.modules['torch.optim'], cfg.optimizer)
    return optimizer


def get_scheduler(cfg):
    scheduler = getattr(sys.modules['torch.optim.lr_scheduler'], cfg.scheduler)
    return scheduler


def get_paths_1_dataset(data_folder, dataset_name):
    paths_folder = os.path.join(data_folder, dataset_name)
    last_number = 0
    paths, _paths = [], []
    if 'MedSeg' in dataset_name:
        for i, name in enumerate(sorted(os.listdir(paths_folder))):
            path = os.path.join(paths_folder, name)
            if i % 5 == 0:
                paths.append(_paths)
                _paths = []
            _paths.append(path)
        paths.append(_paths)
        return paths

    for name in sorted(os.listdir(paths_folder)):
        path = os.path.join(paths_folder, name)
        number_of_patient = int(name.split('_')[0])
        if last_number != number_of_patient:
            paths.append(_paths)
            _paths = []
            last_number = number_of_patient
        _paths.append(path)
    paths.append(_paths)
    return paths


def get_paths(cfg):
    """
    :param cfg: Config
    :return: paths [patient[slice, ...], ...]
    """
    dataset_name = cfg.dataset_name
    paths = []
    if isinstance(cfg.dataset_name, list):
        for name in dataset_name:
            paths.extend(get_paths_1_dataset(cfg.data_folder, name))
        return paths
    paths = get_paths_1_dataset(cfg.data_folder, dataset_name)
    return paths


def show_segmentation(cfg, loader, best_dict, n=1, size=16, device=None):
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Image, Prediction, True mask')
    k = 0
    model = get_model(cfg)(cfg=cfg).to(device)
    model.load_state_dict(torch.load(best_dict))
    for X, y in loader:
        with torch.no_grad():
            X = X.to(device)
            y = y.to(device)

            output = model(X).to(device)
            output = torch.argmax(torch.sigmoid(output), 1).float()
            for i in range(len(X)):
                if len(torch.unique(y[i])) == 1:
                    continue
                plt.subplots(1, 8, figsize=(size, size))
                plt.subplot(1, 3, 1)
                plt.axis('off')
                plt.imshow(X[i].cpu().squeeze(), cmap='gray')
                plt.subplot(1, 3, 2)
                plt.axis('off')
                plt.imshow(output[i].cpu().squeeze(), cmap='gray')
                plt.subplot(1, 3, 3)
                plt.axis('off')
                plt.imshow(y[i].cpu().squeeze(), cmap='gray')
                plt.show()
                k += 1
                if k == n:
                    return
