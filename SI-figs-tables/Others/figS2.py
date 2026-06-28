import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
import random
import time
start = time.time()

df = pd.read_excel('Dataset_I_42.xlsx', header=1, sheet_name=1, index_col=0)
dfx = pd.read_csv('2Dx_mord_A.csv')
lst_0 = ['AATS1s', 'AATSC2c', 'GATS3c', 'GATS3s', 'BCUTv-1l', 'SM1_Dzv', 'RNCG',
         'AETA_alpha', 'ETA_dAlpha_B', 'ETA_dPsi_A', 'nHBAcc', 'VSA_EState3']
X = dfx[lst_0].values
y = df['Guest ratio  (from 1HNMR)'].values

nnn=10000
print("#nnn: ",nnn)
forest = ExtraTreesRegressor(n_estimators=nnn,
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
forest.fit(X, y)
print('SCORE with selected Features: %1.2f' % forest.score(X, y))
print(forest.feature_importances_)
print("")

dfo = pd.read_excel('testdata_9.xlsx', header=1, sheet_name=1, index_col=0)
#testdata_9.xlsx from Dataset_I_12.xlsx
dfv = pd.read_csv('vl_mrd2D.csv')
X_vl = dfv[lst_0].values
y_vl = dfo['Guest ratio  (from 1HNMR)'].values
list_cd = list(range(0,len(y)))

print(y_vl)
print("")
mmm = 6; kkk = 1000
print("#LC start:",kkk)
for i in range(1,7):
    ii = mmm*i
    list_sv0 = []
    list_sv1 = []
    for j in range(kkk):
        lst_rnd = random.sample(list_cd,ii)
        Xc = X[lst_rnd]
        yc = y[lst_rnd]
        forest.fit(Xc,yc)
        y_P1 = forest.predict(X_vl)
        mae_v1 = mean_absolute_error(y_vl,y_P1)
        list_sv1.append(mae_v1)
        y_P0 = forest.predict(Xc)
        mae_0 = mean_absolute_error(yc,y_P0)
        list_sv0.append(mae_0)
    print(ii,round(np.mean(list_sv1),4),round(np.std(list_sv1),4),
          round(np.mean(list_sv0),4),round(np.std(list_sv0),4))
forest.fit(X,y)
y_P1 = forest.predict(X_vl)
mae_v1 = mean_absolute_error(y_vl,y_P1)
y_P0 = forest.predict(X)
mae_0 = mean_absolute_error(y,y_P0)
print(len(y),round(mae_v1,4),0.0,round(mae_0,4),0.0)

print("")
print(time.time()-start)
print("")

