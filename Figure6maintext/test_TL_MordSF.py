import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
from rdkit import Chem
from rdkit.Chem import PandasTools, Descriptors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import LeaveOneOut
import time
start = time.time()

df = pd.read_excel('Dataset_I2_42.xlsx', header=1, sheet_name=1, index_col=0)
dfx = pd.read_csv('2Dx_mord_A.csv')
lst_0 = ['AATS1s', 'AATSC2c', 'GATS3c', 'GATS3s', 'BCUTv-1l', 'SM1_Dzv', 'RNCG',
         'AETA_alpha', 'ETA_dAlpha_B', 'ETA_dPsi_A', 'nHBAcc', 'VSA_EState3']
X = dfx[lst_0].values
y = df['Guest ratio  (from 1HNMR)'].values
model = ExtraTreesRegressor(n_estimators=10000,
                            criterion='squared_error', 
                            max_depth=None, 
                            min_samples_split=2, 
                            min_samples_leaf=1, 
                            min_weight_fraction_leaf=0.0, 
                            max_leaf_nodes=None, 
                            min_impurity_decrease=0.0, 
                            bootstrap=True, 
                            oob_score=False, 
                            n_jobs=None, 
                            random_state=1, 
                            verbose=0, 
                            warm_start=False, 
                            ccp_alpha=0.0, 
                            max_samples=None
                            )
model.fit(X, y)
y_pred = model.predict(X)
print("")
print(mean_squared_error(y, y_pred, squared=False))
print(r2_score(y, y_pred))
print("")

dfo = pd.read_excel('Dataset_I2_12.xlsx', header=1, sheet_name=1, index_col=0)
dfv = pd.read_csv('t_mrd2D.csv')
X_vl = dfv[lst_0].values
y_vl = dfo['Guest ratio  (from 1HNMR)'].values

y_P1 = model.predict(X_vl)
r2s_v1 = r2_score(y_vl, y_P1)
rmse_v1 = mean_squared_error(y_vl, y_P1, squared=False)
mae_v1 = mean_absolute_error(y_vl, y_P1)
print(y_vl)
print([round(y_P1[n], 2) for n in range(len(y_P1))])
print('r2, rmsd, mae',round(r2s_v1,3),round(rmse_v1,3),round(mae_v1,3))
list_ab=abs(y_vl-y_P1)
print(round(np.median(list_ab),3),round(np.mean(list_ab),3),round(np.std(list_ab),3))
print("")
print(time.time()-start)

