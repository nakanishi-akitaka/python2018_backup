# -*- coding: utf-8 -*-
"""
http://yamaguchiyuto.hatenablog.com/entry/python-advent-calendar-2014
Created on Thu Dec  6 15:18:33 2018

@author: Akitaka
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from oreore_ridge import RidgeRegression

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
parameters = {'lamb':np.exp([i for i in range(-30,1)])}
reg = GridSearchCV(RidgeRegression(),parameters,cv=5)
reg.fit(X,y)
best = reg.best_estimator_

""" Plot """
xs = np.linspace(0, 1, 500)
ideal = np.sin(2*np.pi*xs)
plt.plot(xlist,ylist,'bo')
plt.plot(xs,ideal)
plt.plot(xs,np.dot(psi(xs,M),best.coef_))
plt.show()