# -*- coding: utf-8 -*-
"""
1. Regression (k-NN)
2. Reliability (k-NN)
3. Applicability Domain (k-NN)
4. Double Cross Validation

Created on Fri Jul 27 10:06:59 2018

@author: Akitaka
"""

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from time                    import time
from sklearn.datasets        import make_regression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
from my_library              import print_gscv_score, dcv
from my_library              import print_score, yyplot
from sklearn.neighbors       import NearestNeighbors, KNeighborsRegressor

print(__doc__)
start = time()

# サンプルデータの生成
X, y = make_regression(n_samples=100, n_features=2, n_informative=2)
ss = MinMaxScaler()
ss = StandardScaler()
ss.fit(X)
X = ss.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#%%
range_k = np.arange(  3, 11, dtype=int)

param_grid = [{'n_neighbors':range_k}]

print("# Tuning hyper-parameters")
print()
print('Search range')
print('k = ', range_k[0], ' ... ',range_k[len(range_k)-1])
print()

mod = KNeighborsRegressor()
n_splits = 5 
cv = ShuffleSplit(n_splits=n_splits, test_size=0.2)
cv = KFold(n_splits=n_splits, shuffle=True)
gscv = GridSearchCV(mod, param_grid, cv=cv)
gscv.fit(X_train, y_train)
print_gscv_score(gscv)

y_pred = gscv.predict(X_train)
print('train data: ',end="")
print_score(y_train, y_pred)

# step 3. test
y_pred = gscv.predict(X_test)
print('test  data: ',end="")
print_score(y_test,  y_pred)

# step 4. visualize outputs
# yy-plot (train)
y_pred = gscv.predict(X_train)
fig = yyplot(y_train, y_pred)

# yy-plot (test)
y_pred = gscv.predict(X_test)
fig = yyplot(y_test, y_pred)

#%%
# Prediction
y_pred = gscv.predict(X_test)

# Applicability Domain
neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(X_train)
dist = np.mean(neigh.kneighbors(X_test)[0], axis=1)
thr = dist.mean() - dist.std()
y_appd = 2 * (dist > thr) -1

# Standard Deviation (= Uncertainty <-> Reliability)
y_reli = np.std(y_train[neigh.kneighbors(X_test)[1]], axis=1)

results = np.c_[y_test, y_pred, y_reli, y_appd]
columns=['observed y', 'predicted y', 'Std. Dev. ', 'AD']
df = pd.DataFrame(results, columns=columns)
print(df)

#%%
for i in range(10):
    dcv(X, y, mod, param_grid)

print('{:.2f} seconds '.format(time() - start))

