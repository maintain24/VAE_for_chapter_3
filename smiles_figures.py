import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

"""
用来绘制smiles的分子图，结果图保存在同目录下的molecules.png!!!!!!!!!!!!!
"""
# Define the SMILES strings for each group
groups_smiles = {
    # "-CH2COOH": "C(C(=O)O)",
    # "-SO3H": "S(=O)(=O)O",
    # "-COCH3": "C(=O)C",
    # "-CH3": "C"
    # "C12H22O11": "C(C1[C@H](C(C(C(O1)O)O)O)O[C@H]2C(C(C(C(O2)CO)O)O)O)O"  # 纤维素二糖
    # "C5H6O": "C1C(C(C(C(O1)O)O)O)O"  # 纤维素单糖
    # "C5H6O": "C1CCOC1"  # 纤维素吡喃环  C1CCOC1, O1CCCC1, [O]1CCCC1, OC[C@H](O)[C@@H](O)[C@H](O)CO  开环
    # "C5H6O": "OC[C@H](O)[C@@H](O)[C@H](O)CO",
    "esterification acetylation": "CCC(=O)OC[C@H]1OC[C@H](O)[C@@H](O)[C@@H]1O"
    # "C20H38O11": "COCC1[C@H](C(C(C(O1)OC)OC)OC)O[C@H]2C(C(C(C(O2)COC)OC)OC)OC",
    # "Test": "[Co][NH3]3[Eu][Eu][P][Er][Er][Er]"
}

# Create molecule objects from SMILES strings
molecules = {name: Chem.MolFromSmiles(smiles) for name, smiles in groups_smiles.items()}

# Custom drawing options
options = rdMolDraw2D.MolDrawOptions()
options.useBWAtomPalette()  # Use black and white colors for atoms
options.bondLineWidth = 3   # Increase the width of the bonds

# Draw the molecules with custom options
drawer = rdMolDraw2D.MolDraw2DCairo(600, 600)  # Set the size of the image
drawer.SetDrawOptions(options)

# Draw molecules in a grid
drawer.DrawMolecules(list(molecules.values()), legends=list(molecules.keys()))
drawer.FinishDrawing()

# Save to a file or display
with open('molecules.png', 'wb') as f:
    f.write(drawer.GetDrawingText())

