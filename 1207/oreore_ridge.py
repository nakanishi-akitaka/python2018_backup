# -*- coding: utf-8 -*-
"""

http://yamaguchiyuto.hatenablog.com/entry/python-advent-calendar-2014
Created on Thu Dec  6 15:17:36 2018

@author: Akitaka
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class RidgeRegression(BaseEstimator, RegressorMixin):
    def __init__(self,lamb=1.0):
        self.lamb = lamb

    def fit(self,X,y):
        A = np.dot(X.T,X) + self.lamb * np.identity(X.shape[1])
        b = np.dot(X.T,y)
        self.coef_ = np.linalg.solve(A,b)
        return self

    def predict(self,X):
        return np.dot(X,self.coef_)