# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 16:09:24 2018

@author: Akitaka
"""

import numpy as np
from sklearn.model_selection import cross_val_score
from oreore_ridge import RidgeRegression
from knn import KNearestNeighbors_Inheritance

def psi(xlist,M):
    """ make a design matrix """
    ret = []
    for x in xlist:
        ret.append([x**i for i in range(0,M+1)])
    return np.array(ret)

np.random.seed(1)

""" Data """
N = 100
M = 3
xlist = np.linspace(0, 1, N)
ylist = np.sin(2 * np.pi * xlist) + np.random.normal(0, 0.2, xlist.size)
X = psi(xlist,M)
y = ylist

""" Cross validation"""
parameter = {'lamb':0}
reg = RidgeRegression(**parameter)
parameter = {'n_neighbors':1}
reg = KNearestNeighbors_Inheritance(**parameter)
reg.fit(X,y)
y_pred = reg.predict(X)

scores = cross_val_score(reg, X, y, cv=5, scoring='neg_mean_squared_error')
print(scores.mean())
