import numpy as np
import pandas as pd
from dscribe.descriptors import SOAP
from ase import Atoms
from rdkit import Chem
from sklearn.preprocessing import StandardScaler

species = ['Cl', 'F', 'N', 'H', 'Si', 'I', 'C', 'O', 'Br']
r_cut = 100.0#
n_max = 5#
l_max = 4#
soap = SOAP(
    species=species,
    periodic=False,
    r_cut=r_cut,
    n_max=n_max,
    l_max=l_max,
    average="inner",
)
supplier = Chem.SDMolSupplier("for_3d.sdf", removeHs = False)
mols = [mol for mol in supplier]
list_s=[]
for i in range(len(mols)):
    mol = mols[i]
    conf = mol.GetConformer()
    sbl=[]; pos=[]
    for atom in mol.GetAtoms():
        sbl.append(atom.GetSymbol())
        xp=conf.GetAtomPosition(atom.GetIdx()).x
        yp=conf.GetAtomPosition(atom.GetIdx()).y
        zp=conf.GetAtomPosition(atom.GetIdx()).z
        pos.append((xp,yp,zp))
    amol = Atoms(sbl, positions=pos)
    soap_mol = soap.create(amol)
    list_s.append(soap_mol)
df=pd.DataFrame(list_s)
print(df)
lst_0=[]
for i in range(df.shape[1]):
    ltmp=df.iloc[:,i].values
    ltmp=ltmp.reshape(-1,1)
    scaler = StandardScaler().fit(ltmp)
    ltmp = scaler.transform(ltmp)
    scaler_X = StandardScaler().fit(ltmp)
    if scaler_X.var_[0]<1e-5:
        continue
    lst_0.append(i)
dfx=df[lst_0]
print(dfx)
dfx.to_csv('to_soap_gcX.csv')
print("")
for i in range(len(species)):
    for j in range(i,len(species)):
        s1=species[i]; s2=species[j]
        print(s1,s2,soap.get_location((s1,s2)))
print("")

