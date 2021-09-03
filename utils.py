import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt
from custom import metrics, losses, models
from torchvision import models


def set_seed(seed=0xD153A53):
    """
    Sets seed of all modules

    :param seed: seed
    :return: None
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def discretize_segmentation_maps(probs, threshold=0.5):
    threshold = torch.from_numpy(np.array(threshold)).to(probs.device)
    return probs > threshold


def get_metric(cfg):
    """
    cfg.metric

    :param cfg: Config
    :return: metric
    """
    return getattr(sys.modules['custom.metrics'], cfg.metric)


def get_criterion(cfg):
    """
    cfg.criterion

    :param cfg: Config
    :return: loss_function
    """
    return getattr(sys.modules['custom.losses'], cfg.loss_function)


def get_model(cfg):
    """
    cfg.model

    :param cfg: Config
    :return: model
    """
    name = cfg.model
    model = getattr(sys.modules['custom.models'], name)
    return model


def get_optimizer(cfg):
    optimizer = getattr(sys.modules['torch.optim'], cfg.optimizer)
    return optimizer


def get_scheduler(cfg):
    scheduler = getattr(sys.modules['torch.optim.lr_scheduler'], cfg.scheduler)
    return scheduler


def get_paths(cfg):
    """
    :param cfg: Config
    :return: paths [patient[slice, ...], ...]
    """
    paths_folder = os.path.join(cfg.data_folder, cfg.dataset_name)
    last_number = 0
    paths, _paths = [], []
    for name in sorted(os.listdir(paths_folder)):
        path = os.path.join(paths_folder, name)
        number_of_patient = int(name.split('_')[0])
        if last_number != number_of_patient:
            paths.append(_paths)
            _paths = []
            last_number = number_of_patient
        _paths.append(path)
    return paths


def show_segmentation(model, loader, n=1, size=16, threshold=0.5, device=None):
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Image, Prediction, Binarized prediction, True mask')
    k = 0
    for X, y in loader:
        with torch.no_grad():
            X = X.to(device)
            y = y.to(device)

            output = model(X)

            for i in range(len(X)):
                if len(torch.unique(y[i])) == 1:
                    continue
                plt.subplots(1, 8, figsize=(size, size))
                plt.subplot(1, 4, 1)
                plt.axis('off')
                plt.imshow(X[i].cpu().squeeze(), cmap='gray')
                plt.subplot(1, 4, 2)
                plt.axis('off')
                plt.imshow(output[i].cpu().squeeze(), cmap='gray')
                plt.subplot(1, 4, 3)
                plt.axis('off')
                plt.imshow(discretize_segmentation_maps(output[i].cpu().squeeze(), threshold=threshold), cmap='gray')
                plt.subplot(1, 4, 4)
                plt.axis('off')
                plt.imshow(y[i].cpu().squeeze(), cmap='gray')
                plt.show()
                k += 1
                if k == n:
                    return
