from utils import get_model
from data_functions import get_transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import torch
import numpy as np
import nibabel as nib
import random
import string
import os
from config import BinaryModelConfig, MultiModelConfig
from PIL import Image, ImageFont, ImageDraw
from pyunpack import Archive
from inference import window_image


def get_setup():
    # preparing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    transforms = []

    # setup for every model
    for cfg in [BinaryModelConfig, MultiModelConfig]:
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


def read_files(files):
    paths = []
    # creating folder for user
    folder_name = generate_folder_name()
    path = 'images/' + folder_name
    create_folder(path)
    for file in files:
        # saving file from user
        file_path = os.path.join(path, file.name)
        open(file_path, 'wb').write(file.getvalue())

        if file.name.endswith('.rar') or file.name.endswith('.zip'):
            Archive(file_path).extractall(path)
            os.remove(file_path)
            images = []
            # create_folder(rar_path)
            for dcm in os.listdir(path):
                images.append(os.path.join(path, dcm))
            paths.append(images)
        else:
            # Single DICOM
            paths.append([file_path])
    return paths, folder_name


def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
