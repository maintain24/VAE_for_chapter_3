# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

"""
loss曲线起伏太大，将曲线分别进行移动平均、指数平滑、高斯平滑
"""
# Load the dataset
# file_path = r'D:\学习\研三\小论文第二篇（中文）\化工学报\补充数据结果\step_loss\总.xlsx'
file_path = r'/mnt/pycharm_project_VAE/Loss/总.xlsx'
# file_path = r'/mnt/pycharm_project_VAE/Loss/All.csv'
data = pd.read_excel(file_path, engine='openpyxl')
# data = pd.read_csv(file_path)

# Preparing for smoothing
smoothed_data_moving_average = data.copy()
smoothed_data_exponential = data.copy()
smoothed_data_gaussian = data.copy()

# Window size for smoothing (Moving Average and Exponential)
window_size = 10

# Window size for Gaussian smoothing (adjusted to be odd)
window_size_gaussian = 9

# Apply smoothing techniques to each data series (excluding the first column which is the x-axis)
for column in data.columns[1:]:
    # Moving Average
    smoothed_data_moving_average[column] = data[column].rolling(window=window_size, min_periods=1).mean()

    # Exponential Smoothing
    smoothed_data_exponential[column] = data[column].ewm(span=window_size, adjust=False).mean()

    # Gaussian Smoothing (using Savitzky-Golay filter as an approximation)
    smoothed_data_gaussian[column] = savgol_filter(data[column], window_length=window_size_gaussian, polyorder=2)

# Saving the results to new Excel files
smoothed_data_moving_average.to_csv('/mnt/pycharm_project_VAE/Loss/smoothed_data_moving_average.csv', index=False)
smoothed_data_exponential.to_csv('/mnt/pycharm_project_VAE/Loss/smoothed_data_exponential.csv', index=False)
smoothed_data_gaussian.to_csv('/mnt/pycharm_project_VAE/Loss/smoothed_data_gaussian_adjusted.csv', index=False)
