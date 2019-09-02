# -*- coding: utf-8 -*-
"""
  Double Cross Validation
+ Support Vector Machine
+ One-Class Support Vector Machine 
  Regression

Created on Thu Aug  2 15:41:37 2018

@author: Akitaka
"""

import pandas as pd
import numpy as np
from time                    import time
from sklearn.datasets        import make_regression
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm             import SVR
from sklearn.svm             import OneClassSVM
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
# from sklearn.model_selection import cross_val_score
from my_library              import print_gscv_score
from my_library              import print_score_rgr
from my_library              import yyplot
from my_library              import dcv, optimize_gamma

print(__doc__)
start = time()

# サンプルデータの生成
X, y = make_regression(n_samples=100, n_features=2, n_informative=2)
scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#%%
range_c = 2**np.arange(  -5+5, 11, dtype=float)
range_e = 2**np.arange( -10,  1, dtype=float)
range_g = 2**np.arange( -20+20, 11, dtype=float)

param_grid = [
    {'kernel': ['rbf'], 'gamma': range_g,'C': range_c,'epsilon': range_e},
    ]

score = 'accuracy'
print("# Tuning hyper-parameters for {}".format(score))
print()
print('Search range')
print('c = ', range_c[0], ' ... ',range_c[len(range_c)-1])
print('e = ', range_e[0], ' ... ',range_e[len(range_e)-1])
print('g = ', range_g[0], ' ... ',range_g[len(range_g)-1])
print()

mod = SVR()
kf = KFold(n_splits=2, shuffle=True)
gscv = GridSearchCV(mod, param_grid, cv=kf)
gscv.fit(X_train, y_train)
print_gscv_score(gscv)

y_pred = gscv.predict(X_train)
print('train data: ',end="")
print_score_rgr(y_train, y_pred)
y_pred = gscv.predict(X_test)
print('test  data: ',end="")
print_score_rgr(y_test,  y_pred)

y_pred = gscv.predict(X_train)
fig = yyplot(y_train, y_pred)
y_pred = gscv.predict(X_test)
fig = yyplot(y_test, y_pred)

#%%
# Novelty detection by One Class SVM with optimized hyperparameter
clf = OneClassSVM(nu=0.003, kernel=gscv.best_params_['kernel'],
  gamma=gscv.best_params_['gamma'])
clf.fit(X_train)
reliability1 = clf.predict(X_test) # outliers = -1

# Novelty detection by One Class SVM with optimized hyperparameter
optgamma = optimize_gamma(X_train, range_g) 
clf = OneClassSVM(nu=0.003, kernel=gscv.best_params_['kernel'],
  gamma=optgamma)
clf.fit(X_train)
reliability2 = clf.predict(X_test) # outliers = -1

y_pred = gscv.predict(X_test)    # prediction
results = np.c_[y_pred, y_test, reliability1, reliability2]
columns=['predicted y','observed y','reliability1', 'reliability2']
df = pd.DataFrame(results, columns=columns)
print(df)

#%%
for i in range(1):
    dcv(X, y, mod, param_grid)

print('{:.2f} seconds '.format(time() - start))
