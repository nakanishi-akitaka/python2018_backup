# -*- coding: utf-8 -*-
"""
  Double Cross Validation
+ Support Vector Machine
+ One-Class Support Vector Machine 
  Regression

Created on Wed Jul 25 15:51:21 2018

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
#from sklearn.model_selection import cross_val_score
from my_library              import print_gscv_score
from my_library              import print_score
from my_library              import yyplot


def dcv_rgr(mod,param_grid):
    # parameters
    ns_in = 2 # n_splits for inner loop
    ns_ou = 2 # n_splits for outer loop
    
    i = 1 # index of loop
    scores = np.array([]) # list of test scores in outer loop
    kf_ou = KFold(n_splits=ns_ou, shuffle=True)
    
    # [start] outer loop for test of the generalization error
    for train_index, test_index in kf_ou.split(X):
#        start = time()
        X_train, X_test = X[train_index], X[test_index] # inner loop CV
        y_train, y_test = y[train_index], y[test_index] # outer loop 
    
        # [start] inner loop CV for hyper parameter optimization
        kf_in = KFold(n_splits=ns_in, shuffle=True)
        gscv = GridSearchCV(mod, param_grid, cv=kf_in)
        gscv.fit(X_train, y_train)
        # [end] inner loop CV for hyper parameter optimization
        
        # test of the generalization error
        score = gscv.score(X_test, y_test)
        scores = np.append(scores, score)
#        print('dataset: {}/{}  accuracy of inner CV: {:.3f} time: {:.3f} s'.\
#              format(i,ns_ou,score,(time() - start)))
        i+=1
    
    # [end] outer loop for test of the generalization error
    print('  ave, std of accuracy of inner CV: {:.3f} (+/-{:.3f})'\
        .format(scores.mean(), scores.std()*2 ))

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
range_c = 2**np.arange(  -5, 10, dtype=float)
range_g = 2**np.arange( -10,  0, dtype=float)

param_grid = [{'kernel': ['rbf'], 'C':range_c, 'gamma': range_g}]

score = 'accuracy'
print("# Tuning hyper-parameters for {}".format(score))
print()
print('Search range')
print('c = ', range_c[0], ' ... ',range_c[len(range_c)-1])
print('g = ', range_g[0], ' ... ',range_g[len(range_g)-1])
print()

mod = SVR()
ldcv = True
if(ldcv):
    for i in range(10):
        dcv_rgr(mod, param_grid)

kf = KFold(n_splits=2, shuffle=True)
rgr = GridSearchCV(mod, param_grid, cv=kf)
rgr.fit(X_train, y_train)
print_gscv_score(rgr)

y_pred = rgr.predict(X_train)
print('train data: ',end="")
print_score(y_train, y_pred)
y_pred = rgr.predict(X_test)
print('test  data: ',end="")
print_score(y_test,  y_pred)

y_pred = rgr.predict(X_train)
fig = yyplot(y_train, y_pred)
y_pred = rgr.predict(X_test)
fig = yyplot(y_test, y_pred)


#%%

# Novelty detection by One Class SVM with optimized hyperparameter
clf = OneClassSVM(nu=0.10, kernel=rgr.best_params_['kernel'],
  gamma=rgr.best_params_['gamma'])
clf.fit(X_train)

y_pred = rgr.predict(X_test)    # prediction
reliability = clf.predict(X_test) # outliers = -1
results = np.c_[y_pred, y_test, reliability]
columns=['predicted y','observed y','reliability']
df = pd.DataFrame(results, columns=columns)
print(df)

print('{:.2f} seconds '.format(time() - start))

