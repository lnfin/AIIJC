import argparse
import os
import cv2
from production import make_masks, data_to_paths, create_folder, get_setup, make_legend

# parser arguments
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
parser.add_argument("--multi",
                    action="store_true",
                    default=False,
                    help="if \"multi\" shows ground-glass opacities and consolidation")
parser.add_argument("--show_legend",
                    action="store_true",
                    default=False,
                    help="if \"show_legend\" legend shows on image")

# parsing
args = parser.parse_args()
save_folder = args.save_folder

# setup
models, transforms = get_setup()

# reformatting all data to png format and getting paths
paths = data_to_paths(args.data, args.save_folder)

# preparing place for segmentation
for x in ['segmentations', 'annotations']:
    create_folder(os.path.join(save_folder, x))

# prediction
for img, annotation, path in make_masks(paths, models, transforms, args.multi):
    # annotation saving
    print(path)
    name = path.split('\\')[-1].split('.')[0].split('/')[-1]
    print(name)
    with open(os.path.join(save_folder, 'annotations', name + '_annotation.txt'), mode='w') as f:
        f.write(annotation)

    # image saving
    path = os.path.join(save_folder, 'segmentations', name + '_mask.png')
    if args.show_legend:
        img = make_legend(img, annotation)
    cv2.imwrite(path, img)
    print(path, annotation, '', sep='\n')
