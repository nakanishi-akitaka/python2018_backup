# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 13:30:36 2018

@author: Akitaka
"""

from time import time
from sklearn.linear_model    import Ridge, Lasso, ElasticNet
from sklearn.svm             import SVR
from sklearn.neighbors       import KNeighborsRegressor
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, ShuffleSplit, KFold
from sklearn.gaussian_process         import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
from sklearn.ensemble        import GradientBoostingRegressor

#
# functions printing score
#
def print_score(y_test,y_pred):
    rmse  = np.sqrt(mean_squared_error (y_test,y_pred))
    mae   =         mean_absolute_error(y_test,y_pred)
    rmae  = np.sqrt(mean_squared_error (y_test,y_pred))/mae
    r2    =         r2_score           (y_test,y_pred)
    print('RMSE, MAE, RMSE/MAE, R^2 = {:.3f}, {:.3f}, {:.3f}, {:.3f}'\
    .format(rmse, mae, rmae, r2))

def print_gscv_score(gscv):
    print("Best parameters set found on development set:")
    print()
    print(gscv.best_params_)
    print()
#    print("Grid scores on development set:")
#    print()
#    means = gscv.cv_results_['mean_test_score']
#    stds = gscv.cv_results_['std_test_score']
#    for mean, std, params in zip(means, stds, gscv.cv_results_['params']):
#        print("{:.3f} (+/-{:.03f}) for {:}".format(mean, std * 2, params))
       
#}}}


start = time()

ns = 1000 # number of sample

X_train = np.sort(4 * np.random.rand(ns, 1), axis=0)
y_train = 5 * np.random.rand(ns)
# y_train = np.sin(X_train).ravel()

#for i in range(len(y_train)):
#    print(X_train[i],y_train[i])

#
# test data: y = sin(x)
#
X_test = X_train[:]
y_test = np.random.permutation(y_train)

iscaler=1
if(iscaler==1):
    scaler = StandardScaler()
else:
    scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# step 1. model

# step 2. learning
range_f =  0.1*np.arange(  1, 11, dtype=int)
range_n = np.arange( 1,  11, dtype=int)
range_c = 2**np.arange(  -5+10, 11, dtype=float)
range_e = 2**np.arange( -10+5,  1, dtype=float)
range_g = 2**np.arange( -20+20, 11, dtype=float)
range_t = [10, 50, 100, 200, 500]

mod = Ridge()
param_grid = [{'alpha':range_f}]

mod = GradientBoostingRegressor()
param_grid = [{'n_estimators':range_t}]

kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
mod = GaussianProcessRegressor(kernel=kernel)
param_grid = [{'n_restarts_optimizer':range_n}]

mod = RandomForestRegressor()
param_grid = [{'max_features':range_f}]

mod = KNeighborsRegressor()
param_grid = [{'n_neighbors':range_n}]

mod = SVR() 
param_grid = [{'gamma': range_g, 'C': range_c, 'epsilon': range_e}]

n_splits = 3
cv = ShuffleSplit(n_splits=n_splits, test_size=0.2)
cv = KFold(n_splits=n_splits, shuffle=True)
gscv = GridSearchCV(mod, param_grid, cv=cv)
gscv.fit(X_train, y_train)
y_pred = gscv.predict(X_train)
print('train data: ',end="")
print_score(y_train, y_pred)

# step 3. test
y_pred = gscv.predict(X_test)
print('test  data: ',end="")
print_score(y_test,  y_pred)
print('{:.2f} seconds '.format(time() - start))

# step 4. visualize outputs
y_pred = gscv.predict(X_train)
plt.scatter(X_train, y_train,  color='black', label='test data')
plt.plot(X_train, y_pred, color='blue', linewidth=3, label='model')
plt.xlabel('data')
plt.ylabel('target')
plt.legend()
plt.show()
#
#y_pred = gscv.predict(X_test)
#plt.scatter(X_test, y_test,  color='black', label='test data')
#plt.plot(X_test, y_pred, color='blue', linewidth=3, label='model')
#plt.xlabel('data')
#plt.ylabel('target')
#plt.legend()
#plt.show()
