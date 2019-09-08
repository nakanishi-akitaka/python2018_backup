# -*- coding: utf-8 -*-
"""
Example of Machine Learning

1. Regression (Support Vector Machine)
2. Applicability Domain (One-Class Support Vector Machine )
3. Double Cross Validation

Created on Thu Aug  9 10:31:42 2018

@author: Akitaka
"""

import pandas as pd
import numpy as np
from time                    import time
from sklearn.datasets        import make_regression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
from sklearn.svm             import SVR
from my_library              import print_gscv_score_rgr, dcv_rgr, ad_ocsvm

start = time()

X, y = make_regression(n_samples=100, n_features=2, n_informative=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = MinMaxScaler()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVR()

range_c = 2**np.arange(  -5+15, 11, dtype=float)
range_e = 2**np.arange( -10+10,  1, dtype=float)
range_g = 2**np.arange( -20+10, 11, dtype=float)

param_grid = [{'gamma': range_g, 'C': range_c, 'epsilon': range_e}]

cv = ShuffleSplit(n_splits=5, test_size=0.2)
cv = KFold(n_splits=5, shuffle=True)
gscv = GridSearchCV(model, param_grid, cv=cv)
gscv.fit(X_train, y_train)
print_gscv_score_rgr(gscv, X_train, X_test, y_train, y_test, cv)

# Predicted y 
y_pred = gscv.predict(X_test)

# Applicability Domain (inside: +1, outside: -1)
y_appd = ad_ocsvm(X_train, X_test)

results = np.c_[y_pred, y_test, y_appd]
columns=['predicted y','observed y','AD']
df = pd.DataFrame(results, columns=columns)
# print(df[df.AD == 1])
print(df)

if(False):
    dcv_rgr(X, y, model, param_grid, 10)

print('{:.2f} seconds '.format(time() - start))
