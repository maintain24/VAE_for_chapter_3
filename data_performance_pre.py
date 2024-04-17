# -*- coding:utf-8 -*-
import os
import pandas as pd

# 读取xlsx文件
file_path = r'/mnt/pycharm_project_VAE/data/data_smiles.xlsx'
df = pd.read_excel(file_path, engine='openpyxl')

# 删除包含空格的行
df_cleaned = df.dropna(subset=['info.mofid.smiles'])

# 保存处理后的文件
output_path = r'/mnt/pycharm_project_VAE/data/data_smiles_new.csv'
df_cleaned.to_csv(output_path, index=False)

if os.path.exists(output_path):
    print("处理后的文件已成功保存至:", output_path)
else:
    print("保存文件时发生错误，请检查问题。")
