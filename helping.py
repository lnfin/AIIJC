import torch

from production import ProductionCovid19Dataset
import matplotlib.pyplot as plt

# paths = [r'C:\Users\User\Downloads\1.dcm', r'C:\Users\User\Downloads\2.dcm']
paths = [r'C:\Users\User\Downloads\dicoms\56364503.dcm']
ds = ProductionCovid19Dataset(paths)
for X, y in ds:
    X = X.squeeze()
    y = y.squeeze()
    plt.imshow(y, cmap='gray')
    plt.show()
    plt.imshow(X, cmap='gray')
    plt.show()
