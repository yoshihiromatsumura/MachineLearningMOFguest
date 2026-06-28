import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_excel('Dataset_I_42.xlsx', header=1, sheet_name=1, index_col=0)
dfx = pd.read_csv('2Dx_mord_A.csv')
lst_0 = ['AATS1s', 'AATSC2c', 'GATS3c', 'GATS3s', 'BCUTv-1l', 'SM1_Dzv', 'RNCG',
         'AETA_alpha', 'ETA_dAlpha_B', 'ETA_dPsi_A', 'nHBAcc', 'VSA_EState3']
X = dfx[lst_0].values
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
y = df['Guest ratio  (from 1HNMR)'].values

ttt=0
list_h=[]
sary = np.zeros((len(X),len(X)))
for i in range(len(X)):
    xx=sum(X[i]*X[i])
    for j in range(len(X)):
        yy=sum(X[j]*X[j])
        xy=sum(X[i]*X[j])
        sary[i][j]=xy/(xx+yy-xy)
        if j>i:
            list_h.append(sary[i][j])
            if sary[i][j]<=0.0:
                ttt+=1
print(len(list_h),ttt,round(ttt/len(list_h),3),
      round(min(list_h),3),round(max(list_h),3))
fig = plt.figure()
plt.rcParams["font.size"] = 18
ax = fig.add_subplot(1,1,1)
#ax.set(aspect=0.023)
range_bin_width = np.arange(-0.5,1.0,0.01)
ax.hist(list_h, color='blue', bins=range_bin_width)
#ax.set_xlabel('|ddG|')
#plt.xlim(-0.25,3.25)
#plt.ylim(0.0,300.0)
fig.show()
plt.savefig('hist_sim_mx.pdf')

from vendi_score import vendi
#for i in range(len(X)):
#     for j in range(len(X)):
#         if i!=j:
#              sary[i][j]=0.3
print(round(vendi.score_K(sary),3))

