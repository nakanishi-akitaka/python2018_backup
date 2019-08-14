# -*- coding: utf-8 -*-
"""
Double cross validation for classification
Please read README_jp.txt !!!! 

Created on Sat Jul  7 22:40:22 2018

@author: Akitaka
"""
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from time import time
import numpy as np

# parameters
ns_in = 10 # n_splits for inner loop
ns_ou = 10 # n_splits for outer loop

iris = load_iris()
X = iris.data
y = iris.target

i = 1 # index of loop
scores = np.array([]) # list of test scores in outer loop
kf_ou = KFold(n_splits=ns_ou, shuffle=True)

# [start] outer loop for test of the generalization error
for train_index, test_index in kf_ou.split(X):
    start = time()
    X_train, X_test = X[train_index], X[test_index] # inner loop CV
    y_train, y_test = y[train_index], y[test_index] # outer loop 

    # [start] inner loop CV for hyper parameter optimization
    mod = SVC() 
    range_c = 2**np.arange(  0, 3, dtype=float)
    param_grid = [{'kernel': ['rbf'], 'C': range_c,}]
    kf_in = KFold(n_splits=ns_in, shuffle=True)
    gscv = GridSearchCV(mod, param_grid, cv=kf_in, scoring='accuracy')
    gscv.fit(X_train, y_train)
    # [end] inner loop CV for hyper parameter optimization
    
    # test of the generalization error
    y_pred = gscv.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    scores = np.append(scores, score)
    print('outer loop {}'.format(i))
    print('  accuracy = {:.3f}'.format(score))
    print('  {:.3f} seconds '.format(time() - start))
    print()
    i+=1

# [end] outer loop for test of the generalization error
print('  ave, std = {:.3f} (+/-{:.3f})'\
    .format(scores.mean(), scores.std()*2 ))