from utils import get_model
from data_functions import get_transforms
from torch.utils.data import Dataset, DataLoader
import custom.models
import cv2
import torch
import numpy as np
import nibabel as nib
import random
import string
import os


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


def get_folder_name():
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(7)) + '/'


def read_files(files):
    folder_name = get_folder_name()
    path = 'images/' + folder_name
    if not os.path.exists(path):
        os.mkdir(path)
    imgs = list()
    for file in files:
        imgs.append([])
        if file.name.endswith('.nii'):
            nii_path = path + file.name
            open(nii_path, 'wb').write(file.getvalue())
            try:
                images = nib.load(nii_path)
            except:
                return None
            images = np.array(images.dataobj)
            images = np.moveaxis(images, -1, 0)
            os.remove(nii_path)

            for image in images:
                image = window_image(image, -600, 1500)
                image += abs(np.min(image))
                image = image / np.max(image)
                image_path = path + file.name.split('.')[0] + '.png'
                cv2.imwrite(image_path, image * 255)

                imgs[-1].append(image_path)

        else:
            with open(path + file.name, 'wb') as f:
                f.write(file.getvalue())

            imgs[-1].append(path + file.name)
    return imgs, folder_name


def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    return window_image


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

