from utils import get_model
from data_functions import get_transforms
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
from pydicom import dcmread
import pandas as pd


def create_dataframe(stats, mean_annotation):
    df = pd.json_normalize(stats)
    if 'Ground glass' in stats[0]['left lung'].keys():
        df.columns = [
            np.array(["ID", "left lung", "", "right lung", " ", "both", "  "]),
            np.array(
                ["", "Ground glass", "Consolidation", "Ground glass", "Consolidation", "Ground glass",
                 "Consolidation"])
        ]
        df = df.append(pd.Series([
            -1,
            mean_annotation[0][2] / mean_annotation[0][0],
            mean_annotation[0][1] / mean_annotation[0][0],
            mean_annotation[1][2] / mean_annotation[1][0],
            mean_annotation[1][1] / mean_annotation[1][0],
            mean_annotation[0][2] / mean_annotation[0][0] + mean_annotation[1][2] / mean_annotation[1][0],
            mean_annotation[0][1] / mean_annotation[0][0] + mean_annotation[1][1] / mean_annotation[1][0]],
            index=df.columns), ignore_index=True)

        df['ID'] = df['ID'].astype('int32').replace(-1, '3D').astype('str')

        df[["left lung", "", "right lung", " ", "both", "  "]] = df[
            ["left lung", "", "right lung", " ", "both", "  "]].round(1).applymap('{:.1f}'.format)

    else:
        df.columns = np.array(["ID", "left lung", "right lung", "both"])

        df = df.append(pd.Series([
            -1,
            mean_annotation[0][1] / mean_annotation[0][0],
            mean_annotation[1][1] / mean_annotation[1][0],
            mean_annotation[0][1] / mean_annotation[0][0] + mean_annotation[1][1] /
            mean_annotation[1][0]],
            index=df.columns), ignore_index=True)

        df['ID'] = df['ID'].astype('int32').replace(-1, '3D').astype('str')

        df[["left lung", "right lung", "both"]] = df[["left lung", "right lung", "both"]].round(
            1).applymap('{:.1f}'.format)
    return df


def get_statistic(idx, data):
    stat = {'id': idx + 1}
    if 'ground_glass' in data.keys():
        stat['left lung'] = {
            'Ground glass': data['ground_glass'][0],
            'Consolidation': data['consolidation'][0]
        }
        stat['right lung'] = {
            'Ground glass': data['ground_glass'][1],
            'Consolidation': data['consolidation'][1]
        }
        stat['both lungs'] = {
            'Ground glass': sum(data['ground_glass']),
            'Consolidation': sum(data['consolidation'])
        }
    else:
        stat['left lung'] = data['disease'][0]
        stat['right lung'] = data['disease'][1]
        stat['both lungs'] = stat['left lung'] + stat['right lung']
    return stat


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
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(7))


def name_from_filepath(filepath):
    return ''.join(filepath.split('\\')[-1].split('.')[:-1])


class NiftiSaver:
    def __init__(self):
        self.slices = []

    def add(self, slice):
        self.slices.append(slice)

    def save(self, path):
        slices = np.array(self.slices)
        slices_better = []
        for s in slices:
            rgb = np.zeros((s.shape[0], s.shape[1], 1, 1), [('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
            for i in range(s.shape[0]):
                for j in range(s.shape[1]):
                    rgb[i, j] = tuple(s[i, j, :])
            slices_better.append(rgb)
        slices_better = np.array(slices_better)
        ni_img = nib.Nifti1Image(slices_better, np.eye(4))
        nib.save(ni_img, path)


def save_dicom(path, img):
    ds = dcmread(path)

    ds.Rows = img.shape[1]
    ds.Columns = img.shape[0]
    ds.PhotometricInterpretation = "RGB"
    ds.SamplesPerPixel = 3
    ds.BitsStored = 8
    ds.BitsAllocated = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    # ds.PlanarConfiguration = 0
    # ds.is_little_endian = True
    # ds.fix_meta_info()
    ds.PixelData = img.tobytes()

    path = path.replace('images', 'segmentations')
    ds.save_as(path)


def read_web_files(files, user_folder):
    paths = []
    # creating folder for user
    path = os.path.join('images', user_folder)
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
    return paths


def read_files(original_paths, folder):
    # creating folder for user
    paths = []
    for original_path in original_paths:
        # saving file from user
        if original_path.endswith('.rar') or original_path.endswith('.zip'):
            Archive(original_path).extractall(folder)
            image_paths = []
            for img in os.listdir(folder):
                image_paths.append(os.path.join(folder, img))
            paths.append(image_paths)
        else:
            # Single DICOM
            paths.append([original_path])
    return paths


def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


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
