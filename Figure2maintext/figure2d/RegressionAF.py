import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pubchempy as pcp
import numpy as np
import collections
from sklearn.model_selection import LeaveOneOut

df = pd.read_excel('Dataset_I_35.xlsx', header=1, sheet_name=1, index_col=0)
dfx = pd.read_csv('tfeats_pc.csv')
X = dfx.iloc[:,1:].values
y = df['Guest ratio  (from 1HNMR)'].values

forest = ExtraTreesRegressor(n_estimators=10000,
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
                             random_state=None, 
                             verbose=0, 
                             warm_start=False, 
                             ccp_alpha=0.0, 
                             max_samples=None
                             )
forest.fit(X, y)
print(forest.score(X, y))
print(forest.feature_importances_)

def _return_std_p(X, trees, pred):
# 4.3.2 of arXiv:1211.0906
    std = np.zeros(X.shape[0])
    list_u = []
    for tree in trees:
        var_tree = tree.tree_.impurity[tree.apply(X)] #sig^2
        mu_tree = tree.predict(X)
        list_u.append(round(*mu_tree,3))
        std += var_tree + mu_tree ** 2.0
    list_u = collections.Counter(list_u)
    list_u = list_u.most_common()
    std /= len(trees)
    std -= pred ** 2.0
    std[std < 0.0] = 0.0
    std = std ** 0.5
    return std, list_u

def _lu_out(y0,y1,y2,y3,list_u):
    ltmp = pd.read_csv('file_subst').values
    labels = [str(j[0]) for j in ltmp]
    lbset = []
    for i in range(len(y0)):
        if y0[i] == y1:
            lbset.append(labels[i])
    print(*lbset,round(*y1,2),"(exp.)",round(*y2,2),"(pred.)",round(*y3,2),"(dev.)")
    list_u2 = []
    for i in range(len(list_u)):
        lbset2 = []
        for j in range(len(y0)):
            if y0[j] == list_u[i][0]:
                lbset2.append(labels[j])
        list_u2.append((*lbset2,list_u[i]))
    print(list_u2)

print("")
loo = LeaveOneOut()
list_yt = []
list_yp = []
list_yv = []
list_ab = []
XXX = X
for train_index, test_index in loo.split(XXX):
    X_train, X_test = XXX[train_index], XXX[test_index]
    y_train, y_test = y[train_index], y[test_index]
    forest.fit(X_train, y_train)
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)
    y_var, list_uu = _return_std_p(X_test, forest.estimators_, y_test_pred)
    list_yt.append(*y_test)
    list_yp.append(*y_test_pred)
    list_yv.append(*y_var)
    ttt=abs(y_test[0]-y_test_pred[0])
    list_ab.append(ttt)
    _lu_out(y,y_test,y_test_pred,y_var,list_uu)
    print("")
rmse_v = round(mean_squared_error(list_yt, list_yp, squared=False),3)
r2s_v = round(r2_score(list_yt, list_yp),3)
mae_v = round(mean_absolute_error(list_yt, list_yp),3)
print(rmse_v,r2s_v,mae_v)
print(round(np.median(list_ab),3),round(np.mean(list_ab),3),round(np.std(list_ab),3))
print("")
fff='fx_feats_pc_r'
with open('abs_'+fff+'.dat', 'w') as f:
    print(*list_ab,file=f)

plt.errorbar(list_yt, list_yp, yerr = list_yv, capsize=4, fmt='o', markersize=5,
             ecolor='gray', elinewidth=0.7, markeredgecolor = "black", color='black')
plt.xlabel('Actual value')
plt.ylabel('Predicred value')
xxx = min(min(list_yp),min(list_yt))
yyy = max(max(list_yp),max(list_yt))
plt.xlim(xxx-0.2,yyy+0.2)
plt.ylim(xxx-0.2,yyy+0.2)
plt.axline((0,0),(1,1),color='black',lw=1)
plt.gca().set_aspect('equal', adjustable='box')
plt.tight_layout()
ltmp = pd.read_csv('file_subst').values
labels = [str(j[0]) for j in ltmp]
for i, label in enumerate(labels):
    plt.text(list_yt[i],list_yp[i],label,size='x-small')
plt.text(xxx-0.1,yyy,'RMSE: '+str(rmse_v),size='small')
plt.text(xxx-0.1,yyy-0.1,'R^2: '+str(r2s_v),size='small')
plt.text(xxx-0.1,yyy-0.2,'MAE: '+str(mae_v),size='small')
plt.show()
plt.savefig("pltex_"+fff+".pdf")

