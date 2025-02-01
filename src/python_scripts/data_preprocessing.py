#!/usr/bin/env python
# coding: utf-8

"""
Created on Tuesday Jan  30 2025

@author: Pratik
"""

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def main():

    df_data = pd.read_excel("datasets/pr_2014_2024_selected_complete_continuous_stations.xlsx")
    df1_melted = df_data.melt(id_vars=["yy", "mm", "dd", "yymmdd"], 
                        var_name="STATION_NUMBER", 
                        value_name="Station_Value")
    df1_melted["STATION_NUMBER"] = pd.to_numeric(df1_melted["STATION_NUMBER"])
    df_loc = pd.read_excel("datasets/ana_pr_2014_2024_selected_complete_continuous_stations.xlsx")
    merged_df = df1_melted.merge(df_loc[["STATION_NUMBER", "LATITUDE", "LONGITUDE"]], 
                                on="STATION_NUMBER", how="left")
    # Normalize LATITUDE and LONGITUDE to 0-1 range
    scaler = MinMaxScaler()
    merged_df[["LATITUDE", "LONGITUDE"]] = scaler.fit_transform(merged_df[["LATITUDE", "LONGITUDE"]])

    # Convert `yymmdd` to datetime and normalize time between 0-1
    merged_df["yymmdd"] = pd.to_datetime(merged_df["yymmdd"])
    merged_df["time_scaled"] = (merged_df["yymmdd"] - merged_df["yymmdd"].min()) / (
        merged_df["yymmdd"].max() - merged_df["yymmdd"].min()
    )
    merged_df = merged_df.dropna(subset=["Station_Value"])
    # Standardize `Station_Value` using (X - mean) / std
    merged_df["Station_Value"] = (merged_df["Station_Value"] - 
                                merged_df["Station_Value"].mean()) / merged_df["Station_Value"].std()
    # Select the final required columns
    final_df = merged_df[["time_scaled", "LATITUDE", "LONGITUDE", "STATION_NUMBER", "Station_Value"]]

    print("preprocessed dataset -------------")
    print(final_df.head())
    # Save the result
    final_df.to_csv("datasets/final_standardized_dataset.csv", index=False)

if __name__ == '__main__':
    main()