import torch
import numpy as np
import os
import pydicom
import nibabel as nib
import cv2

from inference import ProductionCovid19Dataset
import matplotlib.pyplot as plt

img = nib.load(r'C:\Users\User\Pictures\covid19\object.nii')
print(img.shape)
for gray in img.get_data():
    gray = gray.squeeze()
    print(gray.shape)
    print(np.unique(gray))
    plt.imshow(gray, cmap='gray')
    plt.show()
