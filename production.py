from utils import get_model
from data_functions import get_transforms
from torch.utils.data import Dataset, DataLoader
import custom.models
import cv2
import torch
import numpy as np


class ProductionCovid19Dataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform
        self._len = len(paths)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        path = self.paths[index]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        image = torch.from_numpy(np.array([image], dtype=np.float))
        image = image.type(torch.FloatTensor)
        return image, 'None'


def get_predictions(cfg, paths):
    # best_dict потом будет в конфиге
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(cfg)(cfg=cfg).to(device)
    model.load_state_dict(torch.load(cfg.best_dict, map_location=torch.device('cpu')))
    model.eval()
    _, transform = get_transforms(cfg)
    dataset = ProductionCovid19Dataset(paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, drop_last=False)
    for X, _ in dataloader:
        X = X.to(device)
        X = X / torch.max(X)

        with torch.no_grad():
            output = model(X)
            for pred in output:
                pred = pred.squeeze().cpu()
                pred = torch.argmax(pred, 0).float()
                print(torch.unique(pred))
                maximum = torch.max(pred)
                if maximum > 1:
                    pred = pred / maximum
                print(torch.unique(pred))
                yield pred.numpy()


def percents_of_covid19(lung_mask, covid19_mask):
    covid19_mask_binarized = covid19_mask >= 1
    return np.sum(covid19_mask_binarized) / np.sum(lung_mask)
