# -*- coding: utf-8 -*-
"""
Example of Machine Learning - Regression

1. Regression (Random Forest)
2. Applicability Domain (k-NN)
3. Double Cross Validation

Created on Wed Aug  8 10:29:27 2018

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
from my_library              import print_gscv_score_rgr, dcv_rgr, ad_knn

start = time()

X, y = make_regression(n_samples=100, n_features=10, n_informative=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = MinMaxScaler()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mod = RandomForestRegressor()

range_f =  0.1*np.arange(  1, 10, dtype=int)

param_grid = [{'max_features':range_f}]

cv = ShuffleSplit(n_splits=5, test_size=0.2)
cv = KFold(n_splits=5, shuffle=True)
gscv = GridSearchCV(mod, param_grid, cv=cv)
gscv.fit(X_train, y_train)
print_gscv_score_rgr(gscv, X_train, X_test, y_train, y_test, cv)

# Predicted y 
y_pred = gscv.predict(X_test)
# Applicability Domain (inside: +1, outside: -1)
y_appd = ad_knn(X_train, X_test)
results = np.c_[y_test, y_pred, y_appd]
columns=['observed y', 'predicted y', 'AD']
df = pd.DataFrame(results, columns=columns)
print(df[df.AD == 1])

if(False):
    dcv_rgr(X, y, mod, param_grid, 10)

print('{:.2f} seconds '.format(time() - start))
