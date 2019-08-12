# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 13:44:49 2018

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
# modules
# {{{
from __future__ import print_function

from time import time
from sklearn.svm import SVR
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# }}}
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
def print_search_score(grid_search,X_train,y_train): #{{{
    print('search score')
    y_pred = grid_search.predict(X_train)
    scores = cross_val_score(grid_search.best_estimator_,X_train,y_train,cv=5)
    np.set_printoptions(precision=3)
    print('In cross validation with best params = ',grid_search.best_params_)
    print('  scores for each fold = ' , scores)
    print('  ave, std = {:.3f} (+/-{:.3f})'\
    .format(scores.mean(), scores.std()*2 ))
    print('  ave = grid_search.best_score  :', grid_search.best_score_)
#   print('')
#   print('r2_score(y_train,y_pred ) : %.3f' % (r2_score(y_train,y_pred )))
#   print('= grid_search.score : %.3f' % (grid_search.score(X_train,y_train)))
#   print('')
#   print('grid_search.cv_results_ = ',grid_search.cv_results_)
    print('')
    i_best=grid_search.cv_results_['params'].index(grid_search.best_params_)
    score0=grid_search.cv_results_['split0_test_score'][i_best]
    score1=grid_search.cv_results_['split1_test_score'][i_best]
    score2=grid_search.cv_results_['split2_test_score'][i_best]
    score3=grid_search.cv_results_['split3_test_score'][i_best]
    score4=grid_search.cv_results_['split4_test_score'][i_best]
    print('grid_search.cv_results_[params] = ',i_best)
    print('grid_search.cv_results_[split0_test_score] = {:6.3f}'.format(score0))
    print('grid_search.cv_results_[split1_test_score] = {:6.3f}'.format(score1))
    print('grid_search.cv_results_[split2_test_score] = {:6.3f}'.format(score2))
    print('grid_search.cv_results_[split3_test_score] = {:6.3f}'.format(score3))
    print('grid_search.cv_results_[split4_test_score] = {:6.3f}'.format(score4))
    print('  ave = {:.3f}'.format((score0+score1+score2+score3+score4)/5.0))
#   print('')
#   print('grid_search.cv_results_[mean_test_score] = ',grid_search.cv_results_['mean_test_score'])
#   print('')
#   means = grid_search.cv_results_['mean_test_score']
#   stds  = grid_search.cv_results_['std_test_score']
#   for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
#       print('%0.3f (+/-%0.03f) for %r' % (mean, std*2, params))
#   or 
#   print('grid_scores :', grid_search.grid_scores_) 
#   NOTE: Don't use grid_scores_
#   The grid_scores_ attribute was deprecated in version 0.18
#    in favor of the more elaborate cv_results_ attribute.
#   The grid_scores_ attribute will not be available from 0.20.
#}}}
#
# training data: y = sin(x) + noise
#
X_train = np.sort(5 * np.random.rand(40, 1), axis=0)
y_train = np.sin(X_train).ravel()
y_train[::5] += 3 * (0.5 - np.random.rand(8))
#
# test data: y = sin(x)
#
X_test = X_train[:]
y_test = np.sin(X_test).ravel()
print(__doc__)
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

# step 2. learning -> score
mod.fit(X_train, y_train)

y_pred = mod.predict(X_train)
print('learning   score: ',end="")
print_score(y_train, y_pred)

# step 3. predict
y_pred = mod.predict(X_test)

# step 4. score
print('prediction score: ',end="")
print_score(y_test,  y_pred)
print('{:.2f} seconds '.format(time() - start))
# }}}
#
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
range_c = [2**j for j in range(-2,1)]
param_grid = [
{'kernel':['rbf'],    'C': range_c}
]
# grid_search = GridSearchCV(mod, param_grid, cv=5, n_jobs=-1)
grid_search = GridSearchCV(mod, param_grid, cv=5)
grid_search.fit(X_train, y_train)

y_pred = grid_search.predict(X_train)
print_search_score(grid_search,X_train,y_train)
print('learning   score: ',end="")
print_score(y_train, y_pred)

# step 3. predict
y_pred = grid_search.predict(X_test)

# step 4. score
print('prediction score: ',end="")
print_score(y_test,  y_pred)
# print(grid_search.cv_results_)
print('{:.2f} seconds '.format(time() - start))
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

print(__doc__)

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
