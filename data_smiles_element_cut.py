import pandas as pd
from rdkit import Chem
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# 修改这里的路径为您的CSV文件路径
file_path = 'D:/software/PythonProject/Python/Molecular_VAE_Pytorch-master/data/train.csv'

# 读取CSV文件
data = pd.read_csv(file_path)

# 只保留SMILES列非nan的行
data = data[pd.notna(data['SMILES'])]

# 计算每个化合物的元素种类数量
def count_element_types(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return len(set(atom.GetSymbol() for atom in mol.GetAtoms()))
    return None

data['ElementTypes'] = data['SMILES'].apply(count_element_types)

# 移除无法解析的SMILES
data = data.dropna(subset=['ElementTypes'])

# 绘制元素种类数量的分布直方图
plt.figure(figsize=(10, 6))
plt.hist(data['ElementTypes'], bins=range(min(data['ElementTypes'].astype(int)), max(data['ElementTypes'].astype(int)) + 1), alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribution of Element Types in SMILES', fontsize=18, fontname='Times New Roman')
plt.xlabel('Number of Element Types', fontsize=14, fontname='Times New Roman')
plt.ylabel('Frequency', fontsize=14, fontname='Times New Roman')
plt.xticks(fontsize=12, fontname='Times New Roman')
plt.yticks(fontsize=12, fontname='Times New Roman')
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.grid(axis='y', alpha=0.75)
plt.show()

# 保存更新了 'ElementTypes' 列的原数据框回CSV文件
data.to_csv(file_path, index=False)
