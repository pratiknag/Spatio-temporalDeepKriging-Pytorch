#!/usr/bin/env python
# coding: utf-8

"""
Created on Tuesday Jan  30 2025

@author: Pratik
"""

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

def main():
    T = np.linspace(0, 1, 4000)
    latitudes = np.linspace(55, 45, 64)  # Latitude range from 48.5 to 56.5 with 100 points
    longitudes = np.linspace(15, 25, 64)   # Longitude range from 16 to 37.5 with 100 points
    N = 64*64

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
    
    df_data = pd.read_excel("datasets/pr_2014_2024_selected_complete_continuous_stations.xlsx")
    df_data["yymmdd"] = pd.to_datetime(df_data["yymmdd"])
    df1_melted = df_data.melt(id_vars=["yy", "mm", "dd", "yymmdd"], 
                        var_name="STATION_NUMBER", 
                        value_name="Station_Value")
    df1_melted["STATION_NUMBER"] = pd.to_numeric(df1_melted["STATION_NUMBER"])
    df_loc = pd.read_excel("datasets/ana_pr_2014_2024_selected_complete_continuous_stations.xlsx")
    merged_df = df1_melted.merge(df_loc[["STATION_NUMBER", "LATITUDE", "LONGITUDE"]], 
                                on="STATION_NUMBER", how="left")
    # Normalize LATITUDE and LONGITUDE to 0-1 range
    scaler = MinMaxScaler()
    scaler.fit(merged_df[["LATITUDE", "LONGITUDE"]])
    df[["LATITUDE", "LONGITUDE"]] = scaler.transform(df[["LATITUDE", "LONGITUDE"]])
    print("!!! initial dataset loaded and standardization of locations are completed !!!")
    s_2d = np.vstack((df["LATITUDE"],df["LONGITUDE"])).T
    del(merged_df)
    del(df_data)
    del(df1_melted)
    del(df_loc)
    del(df)
    num_basis_space = [9**2, 17**2, 35**2, 73**2]
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
    idx_zero = np.load("datasets/index_zero.npy")
    idx_zero = idx_zero - 1400
    print(phi.shape)
    phi1 = np.delete(phi,idx_zero,1)
    # zero_counts = np.sum(phi == 0, axis=0)
    # # Get indices of the top 30 columns with the most zeros
    # top_30_indices = np.argsort(zero_counts)[-40:]
    # # Remove these columns from the array
    # phi = np.delete(phi, top_30_indices, axis=1)
    print("!!! calculation of spatial basis functions are completed !!!")
    model = DeepKrigingModel(input_dim=6933)
    model.load_state_dict(torch.load('models/model_interpolation.pth'))
    print("grid generation for each time point starting ######")
    time_points = 6
    num_data = 4000 - time_points
    median_array = np.empty((num_data, 64, 64,time_points))
    ub_array = np.empty((num_data, 64, 64,time_points))
    lb_array = np.empty((num_data, 64, 64,time_points))
    def gen_at_T(i):
        t = T[i]
        num_basis = [50, 350, 1000]
        std_arr = [0.2, 0.09, 0.009]
        mu_knots = [np.linspace(0, 1, int(i)) for i in num_basis]
        phi_t_row = np.zeros((1, sum(num_basis)))
        K = 0
        for res in range(len(num_basis)):
            std = std_arr[res]
            for u in range(num_basis[res]):
                d = np.square(np.absolute(t-mu_knots[res][u]))
                if d >= 0 and d <= 1:
                    phi_t_row[0,u + K] = np.exp(-0.5 * d/(std**2))
                else:
                    phi_t_row[0,u + K] = 0
            K = K + num_basis[res]
        phi_t = np.tile(phi_t_row, (N, 1))
        phi_combined = np.hstack((phi_t,phi1))
        x_tensor = torch.tensor(phi_combined, dtype=torch.float32)
        test_loader = DataLoader(x_tensor, batch_size=N, shuffle=False)
        for inputs in test_loader:
            x1, x2, x3 = model(inputs)
        x1 = x1.detach().numpy()
        x2 = x2.detach().numpy()
        x3 = x3.detach().numpy()
        x1 = x1.reshape(64,64)
        x2 = x2.reshape(64,64)
        x3 = x3.reshape(64,64)
        return x1,x2,x3
    for i in tqdm(range(num_data)):
        # T = 1
        x1,x2,x3 = gen_at_T(i)
        median_array[i,:,:,0] = x1
        ub_array[i,:,:,0] = x2
        lb_array[i,:,:,0] = x3
        
        # T = 2
        x1,x2,x3 = gen_at_T(i+1)
        median_array[i,:,:,1] = x1
        ub_array[i,:,:,1] = x2
        lb_array[i,:,:,1] = x3
        
        # T = 3
        x1,x2,x3 = gen_at_T(i+2)
        median_array[i,:,:,2] = x1
        ub_array[i,:,:,2] = x2
        lb_array[i,:,:,2] = x3
        
        # T = 4
        x1,x2,x3 = gen_at_T(i+3)
        median_array[i,:,:,3] = x1
        ub_array[i,:,:,3] = x2
        lb_array[i,:,:,3] = x3
        
        # T = 5
        x1,x2,x3 = gen_at_T(i+4)
        median_array[i,:,:,4] = x1
        ub_array[i,:,:,4] = x2
        lb_array[i,:,:,4] = x3
        
        # T = 6
        x1,x2,x3 = gen_at_T(i+5)
        median_array[i,:,:,5] = x1
        ub_array[i,:,:,5] = x2
        lb_array[i,:,:,5] = x3
    
    output_file = 'datasets/median_data.npy'
    np.save(output_file, median_array)
    
    output_file = 'datasets/ub_data.npy'
    np.save(output_file, ub_array)
    
    output_file = 'datasets/lb_data.npy'
    np.save(output_file, lb_array)
        
if __name__ == '__main__':
    main()
