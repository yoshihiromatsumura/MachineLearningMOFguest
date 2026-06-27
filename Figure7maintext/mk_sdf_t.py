import pandas as pd
import pubchempy as pcp
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, Descriptors, PandasTools, Descriptors3D
from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import MMFFHasAllMoleculeParams, MMFFOptimizeMolecule

df = pd.read_excel('Dataset_X_12.xlsx', header=1, sheet_name=1, index_col=0)
dfc = df[['CAS No.']].values
properties = ['ConnectivitySMILES']
infos = []
for i in dfc:
    info = pcp.get_properties(properties, i, 'name')
    infos.append(info)
smileses = []
for i in np.arange(len(infos)):
    try:
        smiles = infos[i][0]["ConnectivitySMILES"]
        smileses.append(smiles)
    except:
        smiles = 'NaN'
        smileses.append(smiles)

writer = Chem.SDWriter("for_3d_t.sdf")

for i in smileses:
    mol = Chem.AddHs(Chem.MolFromSmiles(i))
    params = ETKDGv3()
    params.randomSeed = 1
    EmbedMolecule(mol, params)
    MMFFOptimizeMolecule(mol)
    writer.write(mol)

writer.close()

