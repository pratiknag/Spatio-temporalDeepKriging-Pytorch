import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from ST_interpolation import DeepKrigingModel
from tqdm import tqdm


latitudes = np.linspace(0,1, 20)  # Latitude range from 48.5 to 56.5 with 100 points
longitudes = np.linspace(0,1, 20)   # Longitude range from 16 to 37.5 with 100 points
N = 20*20

#Create the meshgrid
longitude_grid, latitude_grid = np.meshgrid(longitudes, latitudes)

# Flatten the grid to get a list of coordinate pairs
latitudes_flat = latitude_grid.flatten()
longitudes_flat = longitude_grid.flatten()
df = pd.DataFrame({
    'LATITUDE': latitudes_flat,
    'LONGITUDE': longitudes_flat
})
## load the training dataset to rescale the long-lat to 0-1
s_2d = np.vstack((df["LATITUDE"],df["LONGITUDE"])).T
num_basis_space = [5**2]
knots_1d_space = [np.linspace(0, 1, int(np.sqrt(i))) for i in num_basis_space]
phi = np.zeros((N, sum(num_basis_space)))    
K_space = 0
for res in range(len(num_basis_space)):
    theta = 1 / np.sqrt(num_basis_space[res]) * 2.5
    knots_s1, knots_s2 = np.meshgrid(knots_1d_space[res], knots_1d_space[res])
    knots = np.column_stack((knots_s1.flatten(), knots_s2.flatten()))
    
    for i in range(num_basis_space[res]):
        d = np.linalg.norm(s_2d - knots[i, :], axis=1) / theta
        for j in range(len(d)):
            if d[j] >= 0 and d[j] <= 1:
                phi[j,i + K_space] = (1-d[j])**6 * (35 * d[j]**2 + 18 * d[j] + 3)/3
            else:
                phi[j,i + K_space] = 0
    K_space += num_basis_space[res]
