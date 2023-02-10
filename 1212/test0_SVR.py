# -*- coding: utf-8 -*-
"""
Example of Machine Learning - Regression

  Double Cross Validation
+ Support Vector Machine
+ One-Class Support Vector Machine 

Created on Fri Aug  3 10:58:18 2018

@author: Akitaka
"""

import numpy as np
from time                    import time
from sklearn.datasets        import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.svm             import SVR
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
from my_library               import print_gscv_score_rgr

start = time()

X, y = make_regression(n_samples=100, n_features=2, n_informative=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = MinMaxScaler()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train[:, None])[:, 0]
y_test = y_scaler.transform(y_test[:, None])[:, 0]


#%%
mod = SVR()

range_c = 2**np.arange(  -5+15, 11, dtype=float)
range_e = 2**np.arange( -10,  1, dtype=float)
range_g = 2**np.arange( -20+10, 11, dtype=float)

param_grid = [
        {'kernel': ['rbf'], 'gamma': range_g,'C': range_c,'epsilon': range_e}]
param_grid = [{'kernel': ['rbf'], 'epsilon': range_e}]
cv = ShuffleSplit(n_splits=5, test_size=0.2)
cv = KFold(n_splits=5, shuffle=True)
gscv = GridSearchCV(mod, param_grid, cv=cv)
gscv.fit(X_train, y_train)

print_gscv_score_rgr(gscv, X_train, X_test, y_train, y_test, cv)

print('{:.2f} seconds '.format(time() - start))