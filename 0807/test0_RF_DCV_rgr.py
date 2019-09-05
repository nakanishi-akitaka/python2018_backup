# -*- coding: utf-8 -*-
"""
Example of Machine Learning - Regression

1. Regression (Random Forest)
2. Double Cross Validation

Created on Tue Aug  7 14:21:37 2018

@author: Akitaka
"""

import pandas as pd
import numpy as np
from time                    import time
from sklearn.datasets        import make_regression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
from sklearn.ensemble        import RandomForestRegressor
from my_library              import print_gscv_score_rgr, dcv_rgr

print(__doc__)

start = time()

X, y = make_regression(n_samples=100, n_features=10, n_informative=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = MinMaxScaler()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%%
mod = RandomForestRegressor()

range_f =  0.1*np.arange(  1, 10, dtype=int)

param_grid = [{'max_features':range_f}]

cv = ShuffleSplit(n_splits=5, test_size=0.2)
cv = KFold(n_splits=5, shuffle=True)
gscv = GridSearchCV(mod, param_grid, cv=cv)
gscv.fit(X_train, y_train)
print_gscv_score_rgr(gscv, X_train, X_test, y_train, y_test, cv)

#%%
#    # Prediction
#    y_pred = gscv.predict(X_test)
#    
#    # Applicability Domain
#    neigh = NearestNeighbors(n_neighbors=5)
#    neigh.fit(X_train)
#    dist = np.mean(neigh.kneighbors(X_test)[0], axis=1)
#    thr = dist.mean() - dist.std()
#    y_appd = 2 * (dist > thr) -1
#    
#    # Standard Deviation (= Uncertainty <-> Reliability)
#    y_reli = np.std(y_train[neigh.kneighbors(X_test)[1]], axis=1)
#    
#    results = np.c_[y_test, y_pred, y_reli, y_appd]
#    columns=['observed y', 'predicted y', 'Std. Dev. ', 'AD']
#    df = pd.DataFrame(results, columns=columns)
#    print(df)

#%%
dcv_rgr(X, y, mod, param_grid, 10)
print('{:.2f} seconds '.format(time() - start))

