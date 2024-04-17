import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import random
"""
数据增强和数据矫正
"""
# 读取XLSX文件
# file_path = r'/mnt/pycharm_project_VAE/smiles_xlsx/tensors_index.xlsx'
file_path = r'D:\学习\研三\小论文第二篇（中文）\化工学报\计算结果.xlsx'
df = pd.read_excel(file_path)

# 提取SMILES列的数据
smiles_column = df['SMILES']


# 定义数据增强函数
def augment_data(smiles_list, num_augmentations=5):
    augmented_data = []
    for smiles in smiles_list:
        molecule = Chem.MolFromSmiles(smiles)

        # 数据增强
        for _ in range(num_augmentations):
            augmented_molecule = Chem.Mol(molecule)
            AllChem.Compute2DCoords(augmented_molecule)
            AllChem.EmbedMolecule(augmented_molecule, randomSeed=random.randint(1, 100))

            # 随机翻转
            if random.choice([True, False]):
                augmented_molecule = Chem.MolFromSmiles(Chem.MolToSmiles(augmented_molecule)[::-1])

            # 添加噪音
            for atom in augmented_molecule.GetAtoms():
                if random.choice([True, False]):
                    atom.SetAtomicNum(random.randint(1, 118))

            augmented_data.append(Chem.MolToSmiles(augmented_molecule))

    return augmented_data


# 定义判断SMILES是否有效的函数
def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and Chem.MolToSmiles(mol) == smiles


# 定义验证和矫正函数
def validate_and_correct_smiles(smiles_list):
    valid_smiles = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)

        # 验证SMILES是否有效
        if mol is not None and Chem.MolToSmiles(mol) == smiles:
            valid_smiles.append(smiles)
        else:
            # 如果SMILES无效，尝试纠正
            corrected_mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if corrected_mol is not None:
                Chem.SanitizeMol(corrected_mol)
                valid_smiles.append(Chem.MolToSmiles(corrected_mol))

    return valid_smiles


# 定义主函数
def main(file_path, augment=True, num_augmentations=5):
    # 读取XLSX文件
    df = pd.read_excel(file_path)

    # 提取SMILES列的数据
    smiles_column = df['SMILES']

    # 数据增强
    if augment:
        augmented_smiles = augment_data(smiles_column, num_augmentations)
        df['augmented_smiles'] = augmented_smiles
    else:
        df['augmented_smiles'] = smiles_column

    # 验证和矫正
    df['validated_and_corrected_smiles'] = validate_and_correct_smiles(df['augmented_smiles'])

    # 保存结果到新的XLSX文件
    # output_file_path = r'/mnt/pycharm_project_VAE/smiles_xlsx/processed_smiles.xlsx'
    output_file_path = r'D:\学习\研三\小论文第二篇（中文）\化工学报\计算结果（修改）.xlsx'
    df.to_excel(output_file_path, index=False)

    # 输出新文件路径
    print("处理后的SMILES序列已保存到:", output_file_path)


# 调用主函数
main(file_path, augment=True, num_augmentations=5)
