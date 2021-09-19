import argparse
from config import Cfg
from production import get_predictions
import os
import nibabel as nib
from production import window_image
import numpy as np
import cv2


def data_to_paths(data):
    local_path = 'images/'
    all_paths = []
    for path in data:
        if not os.path.exists(path):
            print(f'Path \"{path}\" not exists')
            continue
        if path.endswith('.png') or path.endswith('.jpg') or path.endswith('.jpeg'):
            all_paths.append(path)
        elif path.endswith('.nii') or path.endswith('.nii.gz'):
            paths = []
            nii_name = path.split('/')[-1].split('.')[0]
            images = nib.load(path)
            images = np.array(images.dataobj)
            images = np.moveaxis(images, -1, 0)

            for i, image in enumerate(images):
                image = window_image(image, -600, 1500)
                image += abs(np.min(image))
                image = image / np.max(image)
                image_path = os.path.join(local_path, nii_name + str(i) + '.png')
                cv2.imwrite(image_path, image * 255)

                paths.append(image_path)
            all_paths.extend(path)
        else:
            print(f'Path \"{path}\" is not supported format')
    return all_paths


parser = argparse.ArgumentParser()
parser.add_argument("--path",
                    help="image or folder with CT-images,"
                         "supported image formats: png, jpg, jpeg"
                         "also you can use .nii or .nii.gz formats",
                    required=True)
parser.add_argument("--mode",
                    choices=["binary", "multi"],
                    default="binary",
                    help="if \"multi\" shows ground-glass opacities and consolidation")
args = parser.parse_args()

cfg = Cfg(args.mode == "multi")

paths = data_to_paths(args.data)
for pred in get_predictions(cfg, paths):
    print('prediction')
