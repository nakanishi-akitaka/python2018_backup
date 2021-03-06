# -*- coding: utf-8 -*-
"""
Example of y-randammization for kNN, SVR, RF

Created on Tue Oct 16 13:11:13 2018

@author: Akitaka
"""
from time import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print(__doc__)

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

X_train = np.sort(1 * np.pi * np.random.rand(40, 1), axis=0)
y_train = np.sin(X_train).ravel()
y_train[::5] += 3 * (0.5 - np.random.rand(8))
#
# test data: y = sin(x)
#
# X_test = X_train[:]
X_test = np.sort(4 * np.pi * np.random.rand(80, 1), axis=0)
y_test = np.sin(X_test).ravel()

start = time()
print('')
print('')
print('# 1. SVR with default hyper parameters')

# step 1. model
mod = SVR() 

# step 2. learning
y_rand = np.random.permutation(y_train)
mod.fit(X_train, y_rand)
y_pred = mod.predict(X_train)
print_score(y_train,y_pred)

    # step 4. visualize outputs
plt.scatter(X_train, y_rand, color='red', label='train data')
plt.plot(X_train, y_pred, color='black', linewidth=3, label='model')
plt.xlabel('data')
plt.ylabel('target')
plt.legend()
plt.show()

print('{:.2f} seconds '.format(time() - start))

start = time()
print('')
print('')
print('# 2. RF with default hyper parameters')

# step 1. model
from sklearn.ensemble        import RandomForestRegressor
mod = RandomForestRegressor()

# step 2. learning
y_rand = np.random.permutation(y_train)
mod.fit(X_train, y_rand)
y_pred = mod.predict(X_train)
print_score(y_train,y_pred)

# step 4. visualize outputs
plt.scatter(X_train, y_rand, color='red', label='train data')
plt.plot(X_train, y_pred, color='black', linewidth=3, label='model')
plt.xlabel('data')
plt.ylabel('target')
plt.legend()
plt.show()

print('{:.2f} seconds '.format(time() - start))

start = time()
print('')
print('')
print('# 3. kNN with default hyper parameters')

# step 1. model
from sklearn.neighbors       import KNeighborsRegressor
mod = KNeighborsRegressor()

# step 2. learning
y_rand = np.random.permutation(y_train)
mod.fit(X_train, y_rand)
y_pred = mod.predict(X_train)
print_score(y_train,y_pred)

# step 4. visualize outputs
plt.scatter(X_train, y_rand, color='red', label='train data')
plt.plot(X_train, y_pred, color='black', linewidth=3, label='model')
plt.xlabel('data')
plt.ylabel('target')
plt.legend()
plt.show()

print('{:.2f} seconds '.format(time() - start))

