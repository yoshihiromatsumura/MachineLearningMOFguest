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

dfp = pd.read_csv('2D_mord_fA_300_xS.csv')
X_vl =  scaler.transform(dfp[lst_0].values)

list_hh=[]
sary = np.zeros((len(X_vl),len(X)))
for i in range(len(X)):
    xx=sum(X[i]*X[i])
    for j in range(len(X_vl)):
        yy=sum(X_vl[j]*X_vl[j])
        xy=sum(X[i]*X_vl[j])
        sary[j][i]=xy/(xx+yy-xy)
        list_hh.append(sary[j][i])
print(len(list_hh),round(min(list_hh),3),round(max(list_hh),3))
list_h=[];ttt=0;tt2=0
for j in range(len(X_vl)):
    list_h.append(round(max(sary[j]),3))
    if max(sary[j])>0.8:
        ttt+=1
    if max(sary[j])>0.6:
        tt2+=1   
print(len(list_h),round(ttt/len(list_h),3),round(tt2/len(list_h),3),
      round(min(list_h),3),round(max(list_h),3))
fig = plt.figure()
plt.rcParams["font.size"] = 18
ax = fig.add_subplot(1,1,1)
#ax.set(aspect=***)
range_bin_width = np.arange(0.0,1.0,0.01)
ax.hist(list_h, color='blue',bins=range_bin_width)
#plt.xlim(-0.25,3.25)
plt.ylim(0.0,900.0)
fig.show()
plt.savefig('hgn_sim_mx.pdf')
#fig = plt.figure()
#plt.rcParams["font.size"] = 18
#ax = fig.add_subplot(1,1,1)
#range_bin_width = np.arange(-0.5,1.0,0.0005)
#ax.hist(list_hh, color='blue',bins=range_bin_width)
#fig.show()
#plt.savefig('hgn_sim_mx0.pdf')

