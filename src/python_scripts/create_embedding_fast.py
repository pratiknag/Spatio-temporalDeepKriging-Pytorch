#!/usr/bin/env python
# coding: utf-8

"""
Created on Tuesday Jan  30 2025

@author: Pratik
"""

import torch
import numpy as np
import pandas as pd
import csv
from tqdm import tqdm

def main():
    df = pd.read_csv("datasets/dataset-10DAvg.csv")
    print(df.head(3))
    
    s = np.array(df["time_scaled"]).reshape(len(df), 1)
    print(s.shape)
    N_data = len(df)
    
    # Time basis
    num_basis_time = [50, 350, 1000]
    std_arr_time = [0.2, 0.09, 0.009]
    mu_knots_time = [np.linspace(0, 1, int(i)) for i in num_basis_time]
    
    # Space basis
    num_basis_space = [9**2, 17**2, 35**2, 73**2]
    knots_1d_space = [np.linspace(0, 1, int(np.sqrt(i))) for i in num_basis_space]
    
    # Prepare the output CSV file
    # List to collect the rows for phi_t and phi
    data_list = []
    
    # For each row in the dataset, calculate the phi_t and phi and store them
    for j in tqdm(range(N_data)):
        # Calculate time basis (phi_t)
        # Assume s[j, :] is a 1D NumPy array
        s_j = s[j, :].reshape(-1, 1)  # Reshape to enable broadcasting
        phi_t_row = np.zeros((1, sum(num_basis_time)))
        K_time = 0
        for res in range(len(num_basis_time)):
            std = std_arr_time[res]
            mu_knots = np.array(mu_knots_time[res])  # Convert list to NumPy array

            # Compute squared absolute differences in a vectorized way
            d = np.square(np.abs(s_j[:, None] - mu_knots))  # Shape: (len(s_j), num_basis_time[res])

            # Apply the condition (0 ≤ d ≤ 1)
            mask = (d >= 0) & (d <= 1)

            # Compute phi values only where the condition holds
            phi_t_row[0, K_time : K_time + num_basis_time[res]] = np.exp(-0.5 * d / (std**2)) * mask

            K_time += num_basis_time[res]

        # Assuming df, num_basis_space, knots_1d_space are already defined

        phi_row = np.zeros((1, sum(num_basis_space)))
        s_j_2d = np.array([df["LATITUDE"][j], df["LONGITUDE"][j]]).reshape(1, -1)

        K_space = 0
        for res in range(len(num_basis_space)):
            theta = 1 / np.sqrt(num_basis_space[res]) * 2.5

            # Create 2D meshgrid of knots
            knots_s1, knots_s2 = np.meshgrid(knots_1d_space[res], knots_1d_space[res])
            knots = np.column_stack((knots_s1.flatten(), knots_s2.flatten()))  # Shape: (num_knots, 2)

            # Compute Euclidean distance in a vectorized way
            d = np.linalg.norm(s_j_2d - knots, axis=1) / theta  # Shape: (num_knots,)

            # Apply condition (0 ≤ d ≤ 1) using a boolean mask
            mask = (d >= 0) & (d <= 1)

            # Compute the basis function values using vectorized operations
            phi_values = ((1 - d) ** 6 * (35 * d ** 2 + 18 * d + 3) / 3) * mask

            # Assign the computed values to the appropriate slice of phi_row
            phi_row[0, K_space : K_space + num_basis_space[res]] = phi_values

            K_space += num_basis_space[res]


    
        # Combine both phi_t_row and phi_row into one row
        combined_row = np.hstack((phi_t_row,phi_row))
        # print(combined_row)
        # Convert the filtered row to float16
        filtered_row_float16 = np.array(combined_row, dtype=np.float16)
    
        # Append the row to the data list
        data_list.append(filtered_row_float16)
    # Convert the list of rows into a NumPy array
    data_array = np.vstack(data_list)
    del(data_list)
    print("done till here")
    ## Romove the all-zero columns
    # idx_zero = np.array([], dtype=int)
    # for i in tqdm(range(data_array.shape[1])):
    #     if sum(data_array[:,i]!=0)==0:
    #         idx_zero = np.append(idx_zero,int(i))

    idx_zero = np.where(np.all(data_array == 0, axis=0))[0]
    
    phi_reduce = np.delete(data_array,idx_zero,1)
    print(phi_reduce.shape)
    # Save the resulting array as a .npy file
    output_file = 'datasets/phi_float16.npy'
    np.save(output_file, phi_reduce)

    output_file = 'datasets/index_zero.npy'
    np.save(output_file, idx_zero)

    print(f"Data has been written to {output_file}")


if __name__ == '__main__':
    main()