# -*- coding: utf-8 -*-
"""
http://yamaguchiyuto.hatenablog.com/entry/python-advent-calendar-2014
Created on Thu Dec  6 16:10:09 2018

@author: Akitaka
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from knn import KNearestNeighbors_Inheritance

def psi(xlist,M):
    """ make a design matrix """
    ret = []
    for x in xlist:
        ret.append([x**i for i in range(0,M+1)])
    return np.array(ret)

np.random.seed(0)

""" Data for grid search """
N = 10
M = 15
xlist = np.linspace(0, 1, N)
ylist = np.sin(2 * np.pi * xlist) + np.random.normal(0, 0.2, xlist.size)
X = psi(xlist,M)
y = ylist

""" Grid search """
parameters = {'n_neighbors':[i for i in range(1,5)]}
reg = GridSearchCV(KNearestNeighbors_Inheritance(),parameters,cv=5)
reg.fit(X,y)
print(reg.best_params_)
print(reg.best_score_)

""" Plot """
xs = np.linspace(0, 1, 500)
ideal = np.sin(2*np.pi*xs)
y_pred = reg.predict(psi(xs,M))
plt.plot(xlist,ylist,'bo')
plt.plot(xs,ideal)
plt.plot(xs,y_pred)
plt.show()