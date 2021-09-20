import argparse
import os
import cv2
from production import make_masks, data_to_paths, create_folder, get_setup


parser = argparse.ArgumentParser()
parser.add_argument("--data",
                    help="image or folder with CT-images,"
                         "supported image formats: png, jpg, jpeg"
                         "also you can use .nii or .nii.gz formats",
                    required=True)
parser.add_argument("--save_folder",
                    help="folder to save segmentations and images",
                    default="example",
                    required=True)
parser.add_argument("--mode",
                    choices=["binary", "multi"],
                    default="multi",
                    help="if \"multi\" shows ground-glass opacities and consolidation")
args = parser.parse_args()
save_folder = args.save_folder
multi_class = (args.mode == 'multi')
models, transforms = get_setup()
paths = data_to_paths(args.data, args.save_folder)

for x in ['segmentations', 'annotations']:
    create_folder(os.path.join(save_folder, x))

with open(os.path.join(save_folder, 'segmentations_info.txt'), 'w') as f:
    f.write('''> Colors meaning:
    Green - ground-glass opacities
    Red - consolidation
    Aquamarine - ground-glass opacities and/or consolidation (only in binary mode)
    ''')

for img, annotation, path in make_masks(paths, models, transforms, multi_class):
    name = path.split('\\')[-1].split('.')[0]
    with open(os.path.join(save_folder, 'annotations', name + '.txt'), mode='w') as f:
        f.write(annotation)
    path = os.path.join(save_folder, 'segmentations', name + '_mask.png')
    print(path, annotation, '', sep='\n')
    cv2.imwrite(path, img)
