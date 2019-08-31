# -*- coding: utf-8 -*-
"""
Example of Machine Learning - Regression

Created on Wed Aug  1 13:08:37 2018

@author: Akitaka
"""

import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
from time                    import time
from sklearn.datasets        import make_regression
from sklearn.model_selection import train_test_split
from sklearn.svm             import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit, KFold
from my_library              import print_gscv_score, print_gscv_score_rgr
from my_library              import print_score_rgr
from my_library              import yyplot

print(__doc__)
    
start = time()

X, y = make_regression(n_samples=100, n_features=2, n_informative=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

mod = SVR() 

# search range
# range_c = 2**np.arange(  -5  11, dtype=float)
# range_e = 2**np.arange( -10,  1, dtype=float)
# range_g = 2**np.arange( -20, 11, dtype=float)
# 196.29 seconds 
range_c = 2**np.arange(  -5+10, 11, dtype=float)
range_e = 2**np.arange( -10+5,  1, dtype=float)
range_g = 2**np.arange( -20+20, 11, dtype=float)
print()
print('Search range')
print('c = ', range_c[0], ' ... ',range_c[len(range_c)-1])
print('e = ', range_e[0], ' ... ',range_e[len(range_e)-1])
print('g = ', range_g[0], ' ... ',range_g[len(range_g)-1])
print()

param_grid = [
    {'kernel': ['rbf'], 'gamma': range_g,'C': range_c,'epsilon': range_e},
    ]

cv = ShuffleSplit(n_splits=5, test_size=0.2)
cv = KFold(n_splits=5, shuffle=True)
gscv = GridSearchCV(mod, param_grid, cv=cv, scoring='neg_mean_absolute_error')
gscv.fit(X_train, y_train)
print_gscv_score_rgr(gscv, X_train, X_test, y_train, y_test, cv)

#%%
if(False):
    print_gscv_score(gscv)

if(False):
    y_pred = gscv.predict(X_train)
    print('train data: ',end="")
    print_score_rgr(y_train, y_pred)
    fig = yyplot(y_train, y_pred)
    
    y_pred = gscv.predict(X_test)
    print('test  data: ',end="")
    print_score_rgr(y_test,  y_pred)
    fig = yyplot(y_test, y_pred)
    
print('{:.2f} seconds '.format(time() - start))
