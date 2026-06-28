import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

df = pd.read_excel('Dataset_I_42.xlsx', header=1, sheet_name=1, index_col=0)
dfx = pd.read_csv('2Dx_mord_A.csv')
lst_0 = ['AATS1s', 'AATSC2c', 'GATS3c', 'GATS3s', 'BCUTv-1l', 'SM1_Dzv', 'RNCG',
         'AETA_alpha', 'ETA_dAlpha_B', 'ETA_dPsi_A', 'nHBAcc', 'VSA_EState3']
X = dfx[lst_0].values
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
y = df['Guest ratio  (from 1HNMR)'].values

dfo = pd.read_excel('Dataset_I_12.xlsx', header=1, sheet_name=1, index_col=0)
dfv = pd.read_csv('vl_mrd2D.csv')
X_vl = scaler.transform(dfv[lst_0].values)
y_vl = dfo['Guest ratio  (from 1HNMR)'].values

list_h=[]
sary = np.zeros((len(X_vl),len(X)))
for i in range(len(X)):
    xx=sum(X[i]*X[i])
    for j in range(len(X_vl)):
        yy=sum(X_vl[j]*X_vl[j])
        xy=sum(X[i]*X_vl[j])
        sary[j][i]=xy/(xx+yy-xy)
        list_h.append(sary[j][i])
print(round(min(list_h),3),round(max(list_h),3))
fig = plt.figure()
plt.rcParams["font.size"] = 18
ax = fig.add_subplot(1,1,1)
#ax.set(aspect=***)
for i in range(len(X_vl)):
    ax.scatter([i]*len(X),sary[i],alpha=0.6)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlim(-0.5,8.5)
plt.ylim(-1/3.0,1.0)
fig.show()
plt.savefig('pex_sim_mx.pdf')

list_hh=[]
list_hi=[]
for j in range(len(X_vl)):
    list_hh.append(round(max(sary[j]),3))
    list_hi.append(np.argmax(sary[j]))
print(list_hh)
print(list_hi)

