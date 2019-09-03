# -*- coding: utf-8 -*-
"""
Example of Machine Learning - Regression

  Double Cross Validation
+ Support Vector Machine
+ One-Class Support Vector Machine 

Created on Fri Aug  3 10:58:18 2018

@author: Akitaka
"""

import pandas as pd
import numpy as np
from time                    import time
from sklearn.datasets        import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.svm             import SVR, OneClassSVM
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
from my_library              import print_gscv_score_rgr, dcv, optimize_gamma

print(__doc__)

start = time()

X, y = make_regression(n_samples=100, n_features=2, n_informative=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = MinMaxScaler()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%%
mod = SVR()

range_c = 2**np.arange(  -5+15, 11, dtype=float)
range_e = 2**np.arange( -10+10,  1, dtype=float)
range_g = 2**np.arange( -20+10, 11, dtype=float)

param_grid = [
        {'kernel': ['rbf'], 'gamma': range_g,'C': range_c,'epsilon': range_e}]

cv = ShuffleSplit(n_splits=5, test_size=0.2)
cv = KFold(n_splits=5, shuffle=True)
gscv = GridSearchCV(mod, param_grid, cv=cv)
gscv.fit(X_train, y_train)
print_gscv_score_rgr(gscv, X_train, X_test, y_train, y_test, cv)

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
