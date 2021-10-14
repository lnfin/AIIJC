import torch
import numpy as np
import os

from inference import ProductionCovid19Dataset
import matplotlib.pyplot as plt

# paths = [r'C:\Users\User\Downloads\1.dcm', r'C:\Users\User\Downloads\2.dcm']
# paths = [
#     r'C:\Users\User\Downloads\ERIS000011111431\1.2.40.0.13.1.307600158721496850030215119940173642742\1.2.392.200036.9116.2.5.1.37.2418725157.1632389454.745498\1.2.392.200036.9116.2.5.1.37.2418725157.1632389454.748715']
# ds = ProductionCovid19Dataset(paths)
# for X, y in ds:
#     X = X.squeeze()
#     y = y.squeeze()
#     plt.imshow(y, cmap='gray')
#     plt.show()
#     plt.imshow(X, cmap='gray')
#     plt.show()
path = r'C:\Users\User\Downloads\ERIS000011111431\1.2.40.0.13.1.307600158721496850030215119940173642742\1.2.392.200036.9116.2.5.1.37.2418725157.1632389454.745498'
paths = [os.path.join(path, x) for x in os.listdir(path)]
for path in paths:
    if not path.endswith('.dcm'):
        os.rename(path, path + '.dcm')
        path = path + '.dcm'
    print(path)
# paths = [os.path.join(path, x) for x in os.listdir(path)]
