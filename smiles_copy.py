# -*- coding:utf-8 -*-
import pandas as pd

"""
将纳米纤维素csv文件重复10次,数据增强
"""

# Load the provided CSV file
file_path = r'D:/software/PythonProject/Python/Molecular_VAE_Pytorch-master/data/train_CNC.csv'
file_path = r'/mnt/pycharm_project_VAE/data/train_CNC.csv'

# Re-reading the original CSV file to include the header this time
df_with_header = pd.read_csv(file_path)

# Selecting the first 10 rows excluding the header row
data_to_duplicate = df_with_header.iloc[0:10]

# Duplicating the selected rows 10 times and shuffling
duplicated_data = pd.concat([data_to_duplicate] * 100, ignore_index=True)  # 重复100次
duplicated_data_shuffled = duplicated_data.sample(frac=1).reset_index(drop=True)

# Resetting the index to start from 1 instead of 0
duplicated_data_shuffled.index += 1

# Saving the shuffled data to a new CSV file without the index
# output_file_path_updated = 'D:/software/PythonProject/Python/Molecular_VAE_Pytorch-master/data/train_CNC_4.csv'
output_file_path_updated = '/mnt/pycharm_project_VAE/data/train_CNC_4.csv'
duplicated_data_shuffled.to_csv(output_file_path_updated, index=False)


