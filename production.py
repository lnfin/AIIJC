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
from config import BinaryModelConfig, MultiModelConfig, LungsModelConfig


def get_setup():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    transforms = []
    for cfg in [BinaryModelConfig, MultiModelConfig, LungsModelConfig]:
        # getting model
        model = get_model(cfg)(cfg)
        model.load_state_dict(torch.load(cfg.best_dict, map_location=torch.device('cpu')))
        model.to(device)
        model.eval()
        models.append(model)

        # getting transforms
        _, test_transforms = get_transforms(cfg)
        transforms.append(test_transforms)
    return models, transforms


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
            images = nib.load(nii_path)
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


def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_predictions(paths, models, transforms, multi_class=True):
    binary_model, multi_model, lung_model = models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = DataLoader(ProductionCovid19Dataset(paths, transform=transforms[0]), batch_size=1, drop_last=False)
    for X, _ in dataloader:
        X = X.to(device)
        X = X / torch.max(X)

        with torch.no_grad():
            pred = binary_model(X)
            lung = lung_model(X)

            img = X.squeeze().cpu()
            pred = pred.squeeze().cpu()
            pred = torch.argmax(pred, 0).float()
            lung = lung.squeeze().cpu()
            lung = torch.argmax(lung, 0).float()

            if multi_class:
                multi_output = multi_model(X)
                multi_pred = multi_output.squeeze().cpu()
                multi_pred = torch.argmax(multi_pred, 0).float()
                pred = multi_pred * pred
                pred = (pred % 3)
            pred = pred / 2
            yield img.numpy(), pred.numpy(), lung.numpy()


def make_masks(paths, models, transforms, multi_class=True):
    for path, (img, pred, lung) in zip(paths, get_predictions(paths, models, transforms, multi_class)):
        img0 = np.zeros_like(img)
        img1 = (pred == 0.5)
        img2 = (pred == 1)
        img = np.array([img0, img1, img2]) + img * (pred == 0)
        lung_sum = np.sum(lung)
        if lung_sum == 0:  # Костыль-фича
            lung_sum = img.shape[1] * img.shape[2]
            lung_sum = lung_sum / 3.5
            print('lungs not found')
        if multi_class:
            ground_glass = np.sum(img1) / lung_sum
            if ground_glass == np.nan or ground_glass == np.inf:
                ground_glass = 0
            consolidation = np.sum(img2) / lung_sum
            if consolidation == np.nan or consolidation == np.inf:
                consolidation = 0
            annotation = f'Ground-glass opacities - {ground_glass * 100:.1f}%\n' \
                         f'Consolidation - {consolidation * 100:.1f}%'
        else:
            disease = (np.sum(img1) + np.sum(img2)) / lung_sum
            if disease == np.nan or disease == np.inf:
                disease = 0
            annotation = f'Disease - {disease * 100:.1f}%'
            img[0] += (pred == 0.5)
        img = img.swapaxes(0, -1)
        img = np.round(img * 255)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        yield img, annotation, path


def data_to_paths(data, save_folder):
    all_paths = []
    if not os.path.isdir(data):
        data = [data]
    else:
        data = [os.path.join(data, x) for x in os.listdir(data)]
    for path in data:
        print(path)
        if not os.path.exists(path):
            print(f'Path \"{path}\" not exists')
            continue
        if path.endswith('.png') or path.endswith('.jpg') or path.endswith('.jpeg'):
            all_paths.append(path)
        elif path.endswith('.nii') or path.endswith('.nii.gz'):
            if not os.path.exists(os.path.join(save_folder, 'slices')):
                os.mkdir(os.path.join(save_folder, 'slices'))
            paths = []
            nii_name = path.split('\\')[-1].split('.')[0]
            images = nib.load(path)
            images = np.array(images.dataobj)
            images = np.moveaxis(images, -1, 0)

            for i, image in enumerate(images):
                image = window_image(image, -600, 1500)
                image += abs(np.min(image))
                image = image / np.max(image)
                image_path = os.path.join(save_folder, 'slices', nii_name + '_' + str(i) + '.png')
                cv2.imwrite(image_path, image * 255)

                paths.append(image_path)
            all_paths.extend(paths)
        else:
            print(f'Path \"{path}\" is not supported format')
    return all_paths
