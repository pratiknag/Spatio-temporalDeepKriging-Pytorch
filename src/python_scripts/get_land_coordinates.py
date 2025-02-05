#!/usr/bin/env python
# coding: utf-8

"""
Created on Tuesday Jan  30 2025

@author: Pratik
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

def main():


    latitudes = np.linspace(30, 70, 100)  # Latitude range from 48.5 to 56.5 with 100 points
    longitudes = np.linspace(-20, 60, 100)   # Longitude range from 16 to 37.5 with 100 points


    #Create the meshgrid
    longitude_grid, latitude_grid = np.meshgrid(longitudes, latitudes)

    # Flatten the grid to get a list of coordinate pairs
    latitudes_flat = latitude_grid.flatten()
    longitudes_flat = longitude_grid.flatten()

    # Create a DataFrame with latitude and longitude pairs
    df = pd.DataFrame({
        'latitude': latitudes_flat,
        'longitude': longitudes_flat
    })
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"

    world = gpd.read_file(url)

    # Function to check if coordinates are on land
    def is_on_land(lat, lon):
        point = Point(lon, lat)
        return world.contains(point).any()

    # Apply the function to each row of the DataFrame
    df['on_land'] = df.apply(lambda row: is_on_land(row['latitude'], row['longitude']), axis=1)
    print(df.head(20))
    # Filter out rows where the coordinates are on the ocean (i.e., 'on_land' is False)
    df_filtered = df[df['on_land']]

    # Drop the 'on_land' column (optional)
    df_filtered = df_filtered.drop(columns=['on_land'])

    print(df_filtered.head(20))
    print(df_filtered.shape)
    df_filtered.to_csv("datasets/land_coordinates.csv", index=False)
    
if __name__ == '__main__':
    main()
