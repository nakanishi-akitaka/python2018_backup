# -*- coding: utf-8 -*-
"""
ML of Tc by SVM

Created on Wed Jul 18 15:30:33 2018

@author: Akitaka
"""

import pandas as p
import numpy as np
from time                    import time
from matplotlib              import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm             import SVR
from sklearn.metrics         import mean_absolute_error
from sklearn.metrics         import mean_squared_error
from sklearn.metrics         import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit

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
#    print("Grid scores on development set:")
#    print()
#    means = gscv.cv_results_['mean_test_score']
#    stds = gscv.cv_results_['std_test_score']
#    for mean, std, params in zip(means, stds, gscv.cv_results_['params']):
#        print("{:.3f} (+/-{:.03f}) for {:}".format(mean, std * 2, params))

def read_file(name): 
    data = np.array(p.read_csv(filepath_or_buffer=name,
                               header=None,sep=','))[:,:]
    y = data[:,0]
    X = data[:,1:]
    return X, y

start = time()
#
# start of machine learning
#

train_file = 'tc_train.csv'
X, y = read_file(train_file)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# step 1. model
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
range_c = 2**np.arange(   0, 10, dtype=float)
range_e = 2**np.arange(  -5,  0, dtype=float)
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

# n_splits = 2, 5
# https://datachemeng.com/doublecrossvalidation/
# http://univprof.com/archives/16-06-12-3889388.html
cv = ShuffleSplit(n_splits=5)
gscv = GridSearchCV(mod, param_grid, cv=cv, scoring='neg_mean_absolute_error')
gscv.fit(X_train, y_train)
print_gscv_score(gscv)

y_pred = gscv.predict(X_train)
print('train data: ',end="")
print_score(y_train, y_pred)

# step 3. test
y_pred = gscv.predict(X_test)
print('test  data: ',end="")
print_score(y_test,  y_pred)
print('{:.2f} seconds '.format(time() - start))

#%%
# step 4. visualize outputs
# yy-plot (train)
y_pred = gscv.predict(X_train)
plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.title("yy-plot (train)")
plt.scatter(y_train, y_pred)
max_y = np.max(np.array([y_train, y_pred]))
min_y = np.min(np.array([y_train, y_pred]))
ylowlim = min_y - 0.05 * (max_y - min_y)
yupplim = max_y + 0.05 * (max_y - min_y)
plt.plot([ylowlim, yupplim],
         [ylowlim, yupplim],'k-')
plt.ylim( ylowlim, yupplim)
plt.xlim( ylowlim, yupplim)
plt.xlabel("y_observed")
plt.ylabel("y_predicted")

# yy-plot (test)
y_pred = gscv.predict(X_test)
plt.subplot(1,2,2)
plt.title("yy-plot (test)")
plt.scatter(y_test, y_pred)
max_y = np.max(np.array([y_test, y_pred]))
min_y = np.min(np.array([y_test, y_pred]))
ylowlim = ylowlim
yupplim = max_y + 0.05 * (max_y - min_y)
plt.plot([ylowlim, yupplim],
         [ylowlim, yupplim],'k-')
plt.ylim( ylowlim, yupplim)
plt.xlim( ylowlim, yupplim)
plt.xlabel("y_observed")
plt.ylabel("y_predicted")
plt.tight_layout()
plt.show()

# Check: error follows a normal distribution?
# ref:
# http://univprof.com/archives/16-07-20-4857140.html
plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
y_pred = gscv.predict(X_train)
error = np.array(y_pred-y_train)
plt.hist(error)
plt.title("Gaussian? (train)")
plt.xlabel('prediction error')
plt.ylabel('Frequency')

plt.subplot(1,2,2)
y_pred = gscv.predict(X_test)
error = np.array(y_pred-y_test)
plt.hist(error)
plt.title("Gaussian? (test)")
plt.xlabel('prediction error')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
