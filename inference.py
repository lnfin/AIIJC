import argparse
from production import get_predictions
import os
import nibabel as nib
from production import window_image
import numpy as np
import cv2
from production import get_models
from config import BinaryModelConfig, MultiModelConfig, LungsModelConfig
import torch
from production import make_masks, data_to_paths

parser = argparse.ArgumentParser()
parser.add_argument("--data",
                    help="image or folder with CT-images,"
                         "supported image formats: png, jpg, jpeg"
                         "also you can use .nii or .nii.gz formats",
                    required=True)
parser.add_argument("--mode",
                    choices=["binary", "multi"],
                    default="binary",
                    help="if \"multi\" shows ground-glass opacities and consolidation")
parser.add_argument("--save_folder",
                    help="folder to save segmentations and images",
                    default="example")
args = parser.parse_args()
save_folder = args.save_folder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
binary_model, multi_model, lungs_model = get_models()
binary_model = binary_model.to(device)
multi_model = multi_model.to(device)
lungs_model = lungs_model.to(device)

model = None
cfg = BinaryModelConfig
if args.mode == 'multi':
    model = multi_model
    cfg = MultiModelConfig

paths = data_to_paths(args.data, args.save_folder)

if not os.path.exists(os.path.join(save_folder, 'segmentations')):
    os.mkdir(os.path.join(save_folder, 'segmentations'))
if not os.path.exists(os.path.join(save_folder, 'annotations')):
    os.mkdir(os.path.join(save_folder, 'annotations'))
with open(os.path.join(save_folder, 'segmentations_info.txt'), 'w') as f:
    f.write('''> Colors meaning:
    Green - ground-glass opacities
    Red - consolidation
    Aquamarine - ground-glass opacities and/or consolidation (only in binary mode)
    ''')

for img, annotation, path in make_masks(cfg, paths, binary_model, lungs_model,
                                        device, model):
    name = path.split('\\')[-1].split('.')[0]
    with open(os.path.join(save_folder, 'annotations', name + '.txt'), mode='w') as f:
        f.write(annotation)
    path = os.path.join(save_folder, 'segmentations', name + '_mask.png')
    print(path, annotation, '', sep='\n')
    cv2.imwrite(path, img)
