# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 16:00:29 2018

@author: Akitaka
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:set foldmethod=marker:
# command table 
# zo: open  1 step (o->O, all step)
# zc: close 1 step (c->C, all step)
# zr: open  all fold 1 step (r->R, all step)
# zm: close all fold 1 step (m->M, all step)
# PEP8
################################################################################
# 80 characters / 1 line
################################################################################

# ref
# ../0412/test8.py
# https://qiita.com/ishizakiiii/items/0650723cc2b4eef2c1cf
# ../0413/test5.py
# ../0416/test1.py
# ../0417/test1.py
# ../0413/test3.py
# ../0419/test2.py
#
# http://univprof.com/archives/16-05-01-2850729.html
# 1.prepare dataset 
# 2.visualize
# 3.deal with outer sample
# 4.clustering
# 5.learing
# 6.deal with outer sample
# 7.ditermine AD 
# 8.prediction 
 
#
# modules
# {{{
from __future__ import print_function

#import sys
#sys.stdout = open('test0.txt', 'w')
#
from time import time
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
# }}}

print(__doc__)

#
# functions printing score
#
def print_score(y_test,y_pred): #{{{
    rmse  = np.sqrt(mean_squared_error (y_test,y_pred))
    mae   =         mean_absolute_error(y_test,y_pred)
    rmae  = np.sqrt(mean_squared_error (y_test,y_pred))/mae
    r2    =         r2_score           (y_test,y_pred)
    print('RMSE, MAE, RMSE/MAE, R^2 = {:.3f}, {:.3f}, {:.3f}, {:.3f}'\
    .format(rmse, mae, rmae, r2))
#}}}
def print_gscv_score(gscv): #{{{
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
#
# training data: y = sin(x) + noise
#
nn = 8 # noise
nw = 5 # noise step width 
ns = nn * nw # sample

X_train = np.sort(5 * np.random.rand(ns, 1), axis=0)
y_train = np.sin(X_train).ravel()
y_train[::nn] += 3 * (0.5 - np.random.rand(nw))
#
# test data: y = sin(x)
#
X_test = X_train[:]
y_test = np.sin(X_test).ravel()

#
# 1. Basic follow of machine learning
#    SVR with default hyper parameters
#{{{
start = time()
print('')
print('')
print('# 1. SVR with default hyper parameters')

# step 1. model
mod = SVR() 

# step 2. learning
mod.fit(X_train, y_train)
y_pred = mod.predict(X_train)
print('train data: ',end="")
print_score(y_train, y_pred)

# step 3. test
y_pred = mod.predict(X_test)
print('test  data: ',end="")
print_score(y_test,  y_pred)
print('{:.2f} seconds '.format(time() - start))

# step 4. visualize outputs
plt.scatter(X_test, y_test,  color='black', label='test data')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='model')
plt.xlabel('data')
plt.ylabel('target')
plt.legend()
plt.show()

# }}}
#
#%%
# 2. parameter optimization (Grid Search)
#{{{
start = time()
print('')
print('')
print('# 2. parameter optimization (Grid Search)')

# step 1. model
mod = SVR() 

# step 2. learning with optimized parameters
# search range
# search range
# https://datachemeng.com/fastoptsvrhyperparams/
# range_c = 2**np.arange( -5,  10, dtype=float)
# range_e = 2**np.arange( -10,  0, dtype=float)
# range_g = 2**np.arange( -20, 10, dtype=float)
# 1.gamma = miximize gram matrix
# 2.optimize only epsilon with C = 3 (when X is autoscaled) ang opted gamma
# 3.optimize only C with opted epsilon ang opted gamma
# 4.optimize only gamma with opted C and epsilon
range_c = 2**np.arange(  -5, 10, dtype=float)
range_g = 2**np.arange( -10,  0, dtype=float)
range_e = 2**np.arange( -20, 10, dtype=float)
print()
print('Search range')
print('c = ', range_c)
print('g = ', range_g)
print('e = ', range_e)
print()

# Set the parameters by cross-validation
param_grid = [
        {'kernel': ['rbf'], 'gamma': range_g,'C': range_c,'epsilon': range_e},
        ]

from sklearn.model_selection import KFold
cv_kf = KFold(n_splits=2, shuffle=True)
gscv = GridSearchCV(mod, param_grid, cv=cv_kf, scoring='neg_mean_absolute_error')
# gscv = GridSearchCV(mod, param_grid, cv=cv_kf, scoring='r2')
# gscv = GridSearchCV(mod, param_grid, cv=cv_kf, scoring='neg_mean_squared_error')
gscv.fit(X_train, y_train)

y_pred = gscv.predict(X_train)
print_gscv_score(gscv)
print('train data: ',end="")
print_score(y_train, y_pred)

# step 3. test
y_pred = gscv.predict(X_test)
print('test  data: ',end="")
print_score(y_test,  y_pred)
print('{:.2f} seconds '.format(time() - start))

# step 4. visualize outputs
plt.scatter(X_test, y_test,  color='black', label='test data')
plt.plot(X_test, y_pred, color='blue', linewidth=3, label='model')
plt.xlabel('data')
plt.ylabel('target')
plt.legend()
plt.show()
#}}}

#%%
# 
# 3. parameter optimization (Grid Search) for classification
# Parameter estimation using grid search with cross-validation
# http://scikit-learn.org/stable/auto_examples/model_selection/
# plot_grid_search_digits.html
# 20180704test6.py
#{{{

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

# Loading the Digits dataset
digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

score = 'precision'
print("# Tuning hyper-parameters for {}".format(score))
print()

clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='precision_macro')
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("{:.3f} (+/-{:.03f}) for {:}".format(mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_test, y_pred))
print()

#}}}
#sys.stdout = sys.__stdout__
