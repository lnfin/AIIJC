import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pydicom import dcmread


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
    mean_w = int(mean_w / np.sum(lungs))
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


def get_predictions(paths, models, transforms, multi_class=True):
    # preparing
    binary_model, multi_model = models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = DataLoader(ProductionCovid19Dataset(paths, transform=transforms[0]), batch_size=1, drop_last=False)

    # prediction
    for orig_img, img_to_dicom in dataloader:
        orig_img = orig_img.to(device)
        orig_img = orig_img / torch.max(orig_img)

        with torch.no_grad():
            pred = binary_model(orig_img)
            img = orig_img.squeeze().cpu()
            pred = pred.squeeze().cpu()
            pred = torch.argmax(pred, 0).float()

            lung = lung_segmentation(np.array(img), np.array(pred))

            # if multi class we should use both models to predict
            if multi_class and torch.sum(pred) > 0:
                multi_output = multi_model(orig_img)
                multi_pred = multi_output.squeeze().cpu()
                multi_pred = torch.argmax(multi_pred, 0).float()
                multi_pred = (multi_pred % 3)  # model on trained on 3 classes but using only 2
                pred = pred + pred * (multi_pred == 2)  # ground-glass from binary model and consolidation from second
            orig_img = orig_img.squeeze()
            yield img.numpy(), orig_img.numpy(), pred.numpy(), lung, img_to_dicom


def make_masks(paths, models, transforms, multi_class=True):
    def pre_transforms_of_image(img):
        img = img.swapaxes(0, -1)
        img = np.round(img * 255)
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.flip(img, 0)
        return img

    def combine_with_lung(disease, lung, lung_sum):
        disease = disease * lung
        disease_sum = np.sum(disease)
        percents = disease_sum / lung_sum * 100
        return disease, percents, disease_sum

    for path, (img, orig_img, pred, lung, img_to_dicom) in zip(paths,
                                                               get_predictions(paths, models, transforms, multi_class)):
        img_to_dicom = img_to_dicom.squeeze().cpu()
        img_to_dicom = np.array(img_to_dicom, dtype=np.float)
        print(img.dtype, img.shape, img_to_dicom.dtype, img_to_dicom.shape)
        left_lung = lung[0]
        right_lung = lung[1]
        not_disease = (pred == 0)
        left_lung_sum = np.sum(left_lung)
        right_lung_sum = np.sum(right_lung)
        if multi_class:
            consolidation = (pred == 2)  # red channel
            ground_glass = (pred == 1)  # green channel
            img = np.array([np.zeros_like(img), ground_glass, consolidation]) + img * not_disease
            img_to_dicom = np.array(
                [np.zeros_like(img_to_dicom), ground_glass, consolidation]) + img_to_dicom * not_disease
            gg_left, gg_left_percents, gg_left_sum = combine_with_lung(ground_glass, left_lung, left_lung_sum)
            gg_right, gg_right_percents, gg_right_sum = combine_with_lung(ground_glass, right_lung, right_lung_sum)
            cs_left, cs_left_percents, cs_left_sum = combine_with_lung(consolidation, left_lung, left_lung_sum)
            cs_right, cs_right_percents, cs_right_sum = combine_with_lung(consolidation, right_lung, right_lung_sum)
            annotation = {
                'ground_glass': [gg_left_percents, gg_right_percents],
                'consolidation': [cs_left_percents, cs_right_percents]}
            left_data = (left_lung_sum, cs_left_sum, gg_left_sum)
            right_data = (right_lung_sum, cs_right_sum, gg_right_sum)
        else:
            # disease percents
            disease = (pred == 1)
            disease_left, disease_left_percents, disease_left_sum = combine_with_lung(disease, left_lung, left_lung_sum)
            disease_right, disease_right_percents, disease_right_sum = combine_with_lung(disease, right_lung,
                                                                                         right_lung_sum)
            annotation = {
                'disease': [disease_left_percents,
                            disease_right_percents]}
            img = np.array([np.zeros_like(img), disease, disease]) + img * not_disease
            img_to_dicom = np.array([np.zeros_like(img_to_dicom), disease, disease]) + img_to_dicom * not_disease
            left_data = (left_lung_sum, disease_left_sum, 0)
            right_data = (right_lung_sum, disease_right_sum, 0)
        img = pre_transforms_of_image(img)
        img_to_dicom = pre_transforms_of_image(img_to_dicom)
        yield img, orig_img, img_to_dicom, annotation, path, np.array((left_data, right_data))


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
        original_image = torch.from_numpy(np.array([original_image], dtype=np.float))
        original_image = original_image.type(torch.FloatTensor)
        return image, original_image
