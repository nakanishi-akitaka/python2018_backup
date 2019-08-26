# -*- coding: utf-8 -*-
"""
Example of Machine Learning - Regression

Created on Tue Jul 24 11:16:42 2018

@author: Akitaka
"""

# ref
# ../0412/test8.py
# https://qiita.com/ishizakiiii/items/0650723cc2b4eef2c1cf
# ../0413/test5.py
# ../0416/test1.py
# ../0417/test1.py
# ../0413/test3.py
# ../0419/test2.py
#
# ref. follow of machine learning
# http://univprof.com/archives/16-02-11-2849465.html
# http://univprof.com/archives/16-05-01-2850729.html
# https://dev.classmethod.jp/machine-learning/introduction-scikit-learn/
# flow chart of choosing learning method
# http://scikit-learn.org/stable/tutorial/machine_learning_map/
# table of methods
# http://scikit-learn.org/stable/modules/classes.html 
#
# Parameter estimation using grid search with cross-validation
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html

#
# import modules
#
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
from time                    import time
from sklearn.datasets        import make_regression
from sklearn.model_selection import train_test_split
from sklearn.svm             import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from my_library              import print_gscv_score
from my_library              import print_score
from my_library              import yyplot

print(__doc__)
    
start = time()

X, y = make_regression(n_samples=100, n_features=2, n_informative=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

mod = SVR() 

# step 2. learning with optimized parameters
#
# search range
# https://datachemeng.com/supportvectorregression/
# https://datachemeng.com/fastoptsvrhyperparams/
# range_c = 2**np.arange( -5,  10, dtype=float)
# range_e = 2**np.arange( -10,  0, dtype=float)
# range_g = 2**np.arange( -20, 10, dtype=float)
# 1.gamma = miximize gram matrix
# 2.optimize only epsilon with C = 3 (when X is autoscaled) ang opted gamma
# 3.optimize only C with opted epsilon ang opted gamma
# 4.optimize only gamma with opted C and epsilon
range_c = 2**np.arange(  0, 10, dtype=float)
range_e = 2**np.arange( -5,  0, dtype=float)
range_g = 2**np.arange( -10,  5, dtype=float)
print()
print('Search range')
print('c = ', range_c[0], ' ... ',range_c[len(range_c)-1])
print('e = ', range_e[0], ' ... ',range_e[len(range_e)-1])
print('g = ', range_g[0], ' ... ',range_g[len(range_g)-1])
print()

# Set the parameters by cross-validation
param_grid = [
    {'kernel': ['rbf'], 'gamma': range_g,'C': range_c,'epsilon': range_e},
    ]

# ShuffleSplit or KFold(shuffle=True)
# http://scikit-learn.org/0.18/modules/cross_validation.html
# https://mail.google.com/mail/u/0/#sent/QgrcJHsbjCZNCXqKkMlpLbTXWjKWfzHljSl
# https://mail.google.com/mail/u/0/#sent/RdDgqcJHpWcvcDjPgjkjXHLgLnDfdlQzrnZXHZlrxmfB
#
# n_splits = 2, 5
# https://datachemeng.com/doublecrossvalidation/
# http://univprof.com/archives/16-06-12-3889388.html
# n_splits = 2, 5, 10
# https://datachemeng.com/modelvalidation/
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

# estimation for different datasets = OK: MAE, NG: R^2
# http://univprof.com/archives/16-07-04-4453136.html
rgr = GridSearchCV(mod, param_grid, cv=cv, scoring='neg_mean_absolute_error')
rgr.fit(X_train, y_train)
print_gscv_score(rgr)

y_pred = rgr.predict(X_train)
print('train data: ',end="")
print_score(y_train, y_pred)

# step 3. test
y_pred = rgr.predict(X_test)
print('test  data: ',end="")
print_score(y_test,  y_pred)
print('{:.2f} seconds '.format(time() - start))

#%%

# step 4. visualize outputs
# yy-plot (train)
y_pred = rgr.predict(X_train)
fig = yyplot(y_train, y_pred)

# yy-plot (test)
y_pred = rgr.predict(X_test)
fig = yyplot(y_test, y_pred)