import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from matplotlib.ticker import MaxNLocator
"""
这段代码用于根据SMILES的长度区间，划分不同的数据集
"""
# 修改这里的路径为您的CSV文件路径
file_path = 'D:/software/PythonProject/Python/Molecular_VAE_Pytorch-master/data/train.csv'

# 读取CSV文件
data = pd.read_csv(file_path)

# 计算SMILES字符串的长度
data['Length'] = data['SMILES'].str.len()

# 绘制SMILES长度的分布直方图
plt.figure(figsize=(10, 6))
# 确保最大长度是整数
max_length = int(max(data['Length']))
plt.hist(data['Length'], bins=range(0, max_length + 10, 10), alpha=0.7, color='blue', edgecolor='black')

# 设置标题和坐标轴标签，增大字号，改变字体为Times New Roman
plt.title('Distribution of SMILES Lengths', fontsize=28, fontname='Times New Roman')
plt.xlabel('Length of SMILES', fontsize=24, fontname='Times New Roman')
plt.ylabel('Frequency', fontsize=24, fontname='Times New Roman')
# 设置坐标轴字号
plt.xticks(fontsize=20, fontname='Times New Roman')
plt.yticks(fontsize=20, fontname='Times New Roman')
# 使y轴为整数
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
# 显示网格
plt.grid(axis='y', alpha=0.75)
# 展示图形
plt.show()

# 筛选出长度小于10的SMILES字符串及其性能数据
train_1 = data[data['Length'] < 10]

# 筛选出长度在10到20之间的SMILES字符串及其性能数据
train_2 = data[(data['Length'] >= 10) & (data['Length'] < 20)]

# 保存到新的CSV文件
# train_1.to_csv('data/train_1.csv', index=False)
# train_2.to_csv('data/train_2.csv', index=False)

# 保存更新了 'Length' 列的原数据框回CSV文件
data.to_csv(file_path, index=False)
