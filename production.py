from utils import get_model
from data_functions import get_transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import torch
import numpy as np
import nibabel as nib
from pydicom import dcmread
import random
import string
import os
from skimage import io
from sklearn import cluster
from config import BinaryModelConfig, MultiModelConfig, LungsModelConfig
from PIL import Image, ImageFont, ImageDraw
import patoolib


def get_setup():
    # preparing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    transforms = []

    # setup for every model
    for cfg in [BinaryModelConfig, MultiModelConfig, LungsModelConfig]:
        # getting model
        model = get_model(cfg)(cfg)
        model.load_state_dict(torch.load(cfg.best_dict, map_location=device))
        model.eval()
        models.append(model)

        # getting transforms
        _, test_transforms = get_transforms(cfg)
        transforms.append(test_transforms)
    return models, transforms


def generate_folder_name():
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(7)) + '/'


def make_legend(image, annotation):
    # rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    rgb_image = np.round(image).astype(np.uint8)
    image = Image.fromarray(rgb_image)
    old_size = image.size
    if len(annotation.split('\n')) == 3:
        new_size = (old_size[0], old_size[1] + 130)
        new_image = Image.new('RGB', new_size)
        new_image.paste(image)
        font = ImageFont.truetype("arial.ttf", 30)
        draw = ImageDraw.Draw(new_image)
        draw.ellipse((20 + 2, new_size[1] - 30 + 2, 40 - 2, new_size[1] - 10 - 2), fill=(0, 255, 0))
        draw.text((50, new_size[1] - 40),
                  annotation.split('\n')[1], (255, 255, 255), font=font)
        draw.ellipse((20 + 2, new_size[1] - 70 + 2, 40 - 2, new_size[1] - 50 - 2), fill=(0, 0, 255))
        draw.text((50, new_size[1] - 80),
                  annotation.split('\n')[2], (255, 255, 255), font=font)
        draw.text((50, new_size[1] - 120),
                  annotation.split('\n')[0], (255, 255, 255), font=font)
    else:
        new_size = (old_size[0], old_size[1] + 90)
        new_image = Image.new('RGB', new_size)
        new_image.paste(image)
        font = ImageFont.truetype("arial.ttf", 30)
        draw = ImageDraw.Draw(new_image)
        draw.ellipse((20 + 2, new_size[1] - 30 + 2, 40 - 2, new_size[1] - 10 - 2), fill=(0, 255, 255))
        draw.text((50, new_size[1] - 40),
                  annotation.split('\n')[1], (255, 255, 255), font=font)
        draw.text((50, new_size[1] - 80),
                  annotation.split('\n')[0], (255, 255, 255), font=font)
    return np.asarray(new_image)


def data_to_paths(data, save_folder):
    all_paths = []
    create_folder(save_folder)
    if not os.path.isdir(data):  # single file
        data = [data]
    else:  # folder of files
        data = [os.path.join(data, x) for x in os.listdir(data)]

    for path in data:
        if not os.path.exists(path):  # path not exists
            print(f'Path \"{path}\" not exists')
            continue
        # reformatting by type
        if path.endswith('.png') or path.endswith('.jpg') or path.endswith('.jpeg'):
            all_paths.append(path)
        elif path.endswith('.nii') or path.endswith('.nii.gz'):
            # NIftI format will be png format in folder "slices"
            if not os.path.exists(os.path.join(save_folder, 'slices')):
                os.mkdir(os.path.join(save_folder, 'slices'))

            paths = []

            # NIftI to numpy arrays
            nii_name = path.split('\\')[-1].split('.')[0]
            images = nib.load(path)
            images = np.array(images.dataobj)
            images = np.moveaxis(images, -1, 0)

            for i, image in enumerate(images):
                image = window_image(image)  # windowing

                # saving like png image
                image_path = os.path.join(save_folder, 'slices', nii_name + '_' + str(i) + '.png')
                cv2.imwrite(image_path, image * 255)

                paths.append(image_path)
            all_paths.extend(paths)
        else:
            print(f'Path \"{path}\" is not supported format')
    return all_paths


def window_image(image, window_center=-600, window_width=1500):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    window_image += abs(np.min(window_image))
    window_image = window_image / np.max(window_image)
    return window_image


def lung_segmentation(image, disease):
    image = image.copy() * 255
    disease = np.array(disease.copy() * 255, dtype=np.int)
    h, w = image.shape
    mean_w = 0
    pixels = np.sum(image)
    lower = round(pixels / (w * h) * 1.7)
    lower = (min(lower, 180),)
    new_image = image
    upper = (255,)
    thresh = cv2.inRange(new_image, lower, upper)
    thresh = 1 * ((thresh - disease) == 255).astype(np.uint8)
    contours_info = []
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    index = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        contours_info.append((index, area))
        index = index + 1
    contours_info.sort(key=lambda x: x[1], reverse=True)
    lung1 = np.zeros_like(new_image)
    lung2 = np.zeros_like(new_image)
    if len(contours_info) >= 3:
        index_second = contours_info[1][0]
        cv2.drawContours(lung1, [contours[index_second]], 0, (255), -1)
        index_first = contours_info[2][0]
        cv2.drawContours(lung2, [contours[index_first]], 0, (255), -1)
    lungs = lung1 + lung2
    for x, col in enumerate(lungs):
        for y, pixel in enumerate(col):
            mean_w += x * pixel
    mean_w = round(mean_w / np.sum(lungs))
    coef_of_lung_sizes = np.sum(lung1) / np.sum(lung2)
    right = np.zeros_like(new_image)
    left = np.zeros_like(new_image)
    right[:, mean_w] = lungs[:, mean_w]
    left[:, mean_w:] = lungs[:, mean_w:]
    if 0.2 < coef_of_lung_sizes < 5:
        if np.sum(right * lung1) / np.sum(right) > np.sum(right * lung2) / np.sum(right):
            right = lung1
            left = lung2
        else:
            right = lung2
            left = lung1
    return left, right


def read_files(files):
    paths = []
    for file in files:
        # creating folder for user
        folder_name = generate_folder_name()
        path = 'images/' + folder_name
        create_folder(path)

        paths.append([])

        # saving file from user
        file_path = path + file.name
        open(file_path, 'wb').write(file.getvalue())
        # if NIfTI we should get slice
        if file.name.endswith('.dcm'):
            # Single dicom
            paths[-1].append(file_path)

        elif file.name.endswith('.rar'):
            patoolib.extract_archive(file_path, outdir=path)

            images = []
            # create_folder(rar_path)
            for dcm in os.listdir(path):
                if dcm.endswith('.dcm'):
                    paths[-1].append(os.path.join(path, dcm))

        else:
            # Заглузка для теста на пнг
            with open(file_path, 'wb') as f:
                f.write(file.getvalue())

            paths[-1].append(file_path)
            return paths, folder_name

        os.remove(file_path)  # clearing   

    return paths, folder_name


def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_predictions(paths, models, transforms, multi_class=True):
    # preparing
    binary_model, multi_model, lung_model = models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = DataLoader(ProductionCovid19Dataset(paths, transform=transforms[0]), batch_size=1, drop_last=False)

    # prediction
    for X, _ in dataloader:
        X = X.to(device)
        X = X / torch.max(X)

        with torch.no_grad():
            pred = binary_model(X)
            img = X.squeeze().cpu()
            pred = pred.squeeze().cpu()
            pred = torch.argmax(pred, 0).float()
            lung = lung_segmentation(np.array(img), np.array(pred))
            # if multi class we should use both models to predict
            if multi_class and torch.sum(pred) > 0:
                multi_output = multi_model(X)
                multi_pred = multi_output.squeeze().cpu()
                multi_pred = torch.argmax(multi_pred, 0).float()
                multi_pred = (multi_pred % 3)  # model on trained on 3 classes but using only 2
                pred = pred + pred * (multi_pred == 2)  # ground-glass from binary model and consolidation from second
            pred = pred  # to [0;1] range
            yield img.numpy(), pred.numpy(), lung


def make_masks(paths, models, transforms, multi_class=True):
    for path, (img, pred, lung) in zip(paths, get_predictions(paths, models, transforms, multi_class)):
        left = lung[0]
        right = lung[1]
        not_disease = (pred == 0)
        lung_left = np.sum(left)
        lung_right = np.sum(right)
        if multi_class:
            consolidation = (pred == 2)  # red channel
            ground_glass = (pred == 1)  # green channel
            img = np.array([np.zeros_like(img), ground_glass, consolidation]) + img * not_disease
            gg_left = ground_glass * left
            gg_right = ground_glass * right
            cs_left = consolidation * left
            cs_right = consolidation * right
            gg_left_percents = np.sum(gg_left) / lung_left * 100
            gg_right_percents = np.sum(gg_right) / lung_right * 100
            cs_left_percents = np.sum(cs_left) / lung_left * 100
            cs_right_percents = np.sum(cs_right) / lung_right * 100
            annotation = {
                'ground_glass': [gg_left_percents, gg_right_percents],
                'consolidation': [cs_left_percents, cs_right_percents]}
            left_data = (left, cs_left, gg_left)
            right_data = (right, cs_right, gg_left)
        else:
            # disease percents
            disease = (pred == 1)
            disease_left = disease * left
            disease_right = disease * right
            annotation = {
                'disease': [np.sum(disease_left) / lung_left * 100,
                            np.sum(disease_right) / lung_right * 100]}
            img = np.array([np.zeros_like(img), disease, disease]) + img * not_disease
            left_data = (left, disease_left)
            right_data = (right, disease_right)
        img = img.swapaxes(0, -1)
        img = np.round(img * 255)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.flip(img, 0)
        yield img, annotation, path, np.array(left_data, right_data)


class ProductionCovid19Dataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform
        self._len = len(paths)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        path = self.paths[index]
        dicom = dcmread(path)
        original_image = dicom.pixel_array
        try:
            orientation = dicom.ImageOrientationPatient
        except AttributeError:
            orientation = None
        print(orientation)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = window_image(original_image)
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        image = torch.from_numpy(np.array([image], dtype=np.float))
        image = image.type(torch.FloatTensor)
        return image, original_image
