import argparse
import os
import numpy as np
from zipfile import ZipFile
from inference import make_masks
from production import *
from time import time

# argument parser
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

# parsing
args = parser.parse_args()
save_folder = args.save_folder

start_time_1 = time()
# setup
models, transforms = get_setup()
for folder in ['segmentations/', 'images/', 'checkpoints/']:
    create_folder(folder)

start_time_2 = time()
# prediction
user_folder = generate_folder_name()
user_dir = os.path.join('segmentations', user_folder)
create_folder(os.path.join('images', user_folder))
create_folder(user_dir)
create_folder(os.path.join(user_dir, 'segmentations'))
create_folder(os.path.join(user_dir, 'annotations'))
paths = read_files([args.data], os.path.join('images', user_folder))
print(paths)
name = name_from_filepath(args.data)
print(name)
stats = []
zip_path = os.path.join(save_folder, f'segmentations_{name}.zip')
zip_obj = ZipFile(zip_path, 'w')
mean_data = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float64)
start_time_3 = time()
saver = NiftiSaver()
for idx, data in enumerate(make_masks(paths[0], models, transforms, args.multi)):
    img, orig_img, img_to_dicom, annotation, path, _mean_data = data
    mean_data += _mean_data
    img_to_save = img.astype(np.uint8)
    if not path.endswith('.png') and not path.endswith('.jpg') and not path.endswith('.jpeg'):
        saver.add(img)
        print(path)
    stat = get_statistic(idx, annotation)
    stats.append(stat)
nifti_path = os.path.join(save_folder, 'object.nii')
saver.save(nifti_path)
zip_obj.write(nifti_path)
print(time() - start_time_3)
print(time() - start_time_2)
df = create_dataframe(stats, mean_data)
# Save statistics
statistic_path = os.path.join(save_folder, f'statistics_{name}.xlsx')
df.to_excel(statistic_path)
zip_obj.write(nifti_path)
print('Segmentation zip path:', zip_path)
print('Statistic path:', statistic_path)
# Close zip
zip_obj.close()
print(time() - start_time_1)
