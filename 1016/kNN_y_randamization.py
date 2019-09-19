# -*- coding: utf-8 -*-
"""
Example of y-randammization

Created on Fri Oct  12 16:00:00 2018
@author: Akitaka
"""
# Demonstration of y-randomization

from time                    import time

import numpy as np
from sklearn.datasets        import make_regression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
from sklearn.neighbors       import KNeighborsRegressor

from my_library              import print_gscv_score_rgr
from my_library              import y_randamization_rgr

start = time()

# settings
scaler = MinMaxScaler()
scaler = StandardScaler()
range_k = np.arange(  3, 11, dtype=int)
param_grid = [{'n_neighbors':range_k}]
cv = ShuffleSplit(n_splits=5, test_size=0.2)
cv = KFold(n_splits=5, shuffle=True)

# generate sample dataset
X, y = make_regression(n_samples=1000, n_features=4, n_informative=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# autoscaling
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("# modeling and prediction")
model = KNeighborsRegressor()
gscv = GridSearchCV(model, param_grid, cv=cv)
gscv.fit(X_train, y_train)
print_gscv_score_rgr(gscv, X_train, X_test, y_train, y_test, cv)

niter=10
y_randamization_rgr(X_train, y_train, model, param_grid, niter)