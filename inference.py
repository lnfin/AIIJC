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


def data_to_paths(data, save_folder):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    if not os.path.exists(os.path.join(save_folder, 'slices')):
        os.mkdir(os.path.join(save_folder, 'slices'))
    if not os.path.exists(os.path.join(save_folder, 'segmentations')):
        os.mkdir(os.path.join(save_folder, 'segmentations'))
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
                # print(image_path)

                paths.append(image_path)
            all_paths.extend(paths)
        else:
            print(f'Path \"{path}\" is not supported format')
    return all_paths


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        help="Укажите путь до файла или папки с КТ изображениями, "
                            "поддерживаемые форматы: png, jpg, jpeg "
                            "и .nii, .nii.gz",
                        required=True)
    parser.add_argument("--multi",
                        type=bool,
                        default=False,
                        help="Если указан этот параметр, то будет отображены области матового стекла и консолидации")
    parser.add_argument("--save_folder",
                        help="Папка с результатом работы",
                        default="example")
    return parser.parse_args()


if __name__ == "__main__":
    print('Инициализация...')
    args = get_argparser()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    binary_model, multi_model, lungs_model = get_models(device)

    print(args.multi)
    cfg = BinaryModelConfig
    if args.multi:
        cfg = MultiModelConfig

    print('Делаем предсказания...')
    paths = data_to_paths(args.data, args.save_folder)
    for path, (img, pred, lung) in zip(paths, get_predictions(cfg,
                                                            binary_model, lungs_model,
                                                            paths, device,
                                                            multi_model=args.multi)):
        img0 = (lung == 1)
        img1 = (pred == 0.5)
        img2 = (pred == 1)
        img = np.array([img0, img1, img2]) + img * (lung == 0)
        img = img.swapaxes(0, -1)
        img = np.round(img * 255)
        path = os.path.join(args.save_folder, 'segmentations', path.split('\\')[-1][:-4] + '_mask.png')
        print(path)
        print(f'Ground-class opacities - {np.sum(img1) / np.sum(lung) * 100:.1f}%')
        if args.multi:
            print(f'Consolidation - {np.sum(img2) / np.sum(lung) * 100:.1f}%')
        cv2.imwrite(path, img)
        # cv2.imwrite(path[:-4] + '_2.png', pred * 255)
