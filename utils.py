import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt
import custom.models
import custom.metrics
import custom.losses


def set_seed(seed=0xD153A53):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_metric(cfg):
    return getattr(sys.modules['custom.metrics'], cfg.metric)


def get_criterion(cfg):
    return getattr(sys.modules['custom.losses'], cfg.criterion)


def get_model(cfg):
    return getattr(sys.modules['custom.models'], cfg.model)


def get_optimizer(cfg):
    return getattr(sys.modules['torch.optim'], cfg.optimizer)


def get_scheduler(cfg):
    if cfg.scheduler:
        return getattr(sys.modules['torch.optim.lr_scheduler'], cfg.scheduler)
    return FakeScheduler


def get_paths_1_dataset(data_folder, dataset_name):
    paths_folder = os.path.join(data_folder, dataset_name)
    last_number = 0
    paths, _paths = [], []

    # MedSeg has only one NIftI for many patients
    if 'MedSeg' in dataset_name:
        for i, name in enumerate(sorted(os.listdir(paths_folder))):
            path = os.path.join(paths_folder, name)
            if i % 5 == 0:
                paths.append(_paths)
                _paths = []
            _paths.append(path)
        paths.append(_paths)
        return paths

    # adding paths by patients
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
    # if list for adding paths from every dataset
    if isinstance(cfg.dataset_name, list):
        paths = []
        for name in cfg.dataset_name:
            paths.extend(get_paths_1_dataset(cfg.data_folder, name))
        return paths

    # if solo dataset
    paths = get_paths_1_dataset(cfg.data_folder, cfg.dataset_name)
    return paths


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


class FakeScheduler:
    def __init__(self, *args, **kwargs):
        self.lr = kwargs['lr']

    def step(self, *args, **kwargs):
        pass

    def get_last_lr(self, *args, **kwargs):
        return [self.lr]
