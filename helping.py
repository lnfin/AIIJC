import numpy as np
import nibabel as nib
import cv2


def load_nii(image_path: str) -> np.array:
    """Read .nii or .nii.gz scan and return numpy array."""
    image_nii = nib.load(image_path)
    images = np.array(image_nii.dataobj)
    images = np.moveaxis(images, -1, 0)  # [W, H, C] -> [C, H, W]
    return images


def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    return window_image


images = load_nii(r'C:\Users\User\Pictures\covid19\val_im.nii.gz')
for i, image in enumerate(images):
    image = window_image(image, -600, 1500)
    minimum = np.min(image)
    if minimum < 0:
        image -= minimum
    image = image / np.max(image)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(r'C:\Users\User\Pictures\covid19\images\{}.png'.format(str(i)), np.round(image * 255))
