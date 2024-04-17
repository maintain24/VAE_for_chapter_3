# -*- coding:utf-8 -*-
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol

# 从SMILES生成分子对象
smiles = 'C1=CC=CC=C1'  # 例子：苯的SMILES
mol = Chem.MolFromSmiles(smiles)

# 添加氢原子
mol_with_h = Chem.AddHs(mol)

# 生成3D坐标
AllChem.EmbedMolecule(mol_with_h)
AllChem.UFFOptimizeMolecule(mol_with_h)

# 使用Py3Dmol绘制3D结构
view = py3Dmol.view(width=400, height=400)
view.addModel(Chem.MolToMolBlock(mol), 'mol')
view.setStyle({'stick': {}})
view.zoomTo()
view.show()