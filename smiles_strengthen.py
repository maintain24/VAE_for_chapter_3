import pandas as pd
import random
from rdkit import Chem
from rdkit.Chem import AllChem

"""
数据增强，分子式旋转等操作，但所有元素的化合价都不对，该方法不行
"""
def augment_data(smiles, num_augmentations=10):
    augmented_data = []
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return augmented_data

    for _ in range(num_augmentations):
        augmented_molecule = Chem.Mol(molecule)
        if augmented_molecule is None:
            continue

        # 使用不同的随机种子来计算2D坐标
        AllChem.Compute2DCoords(augmented_molecule, randomSeed=random.randint(1, 10000))

        # 生成增强后的SMILES
        smiles_augmented = Chem.MolToSmiles(augmented_molecule)
        if smiles_augmented:
            augmented_data.append(smiles_augmented)

    return augmented_data

# 读取CSV文件
file_path = "D:\\software\\PythonProject\\Python\\Molecular_VAE_Pytorch-master\\data"
df = pd.read_csv(f"{file_path}/train_CNC.csv")

# 应用数据增强并保存增强结果
augmented_smiles = []
augmented_performance = []
for _, row in df.iterrows():
    smiles = row['SMILES']
    performance = row['Performances']
    augmented_smiles.extend(augment_data(smiles, 10))
    augmented_performance.extend([performance] * 10)

# 创建新的DataFrame
augmented_df = pd.DataFrame({'SMILES': augmented_smiles, 'Performances': augmented_performance})

# 保存到新的CSV文件
augmented_df.to_csv(f"{file_path}/train_3.csv", index=False)
