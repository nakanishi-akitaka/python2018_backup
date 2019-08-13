# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 23:21:31 2018

@author: Akitaka
"""
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

kf = KFold(n_splits=4, shuffle=True)
# kf.get_n_splits(X)
# print(kf)  

#t1,t2,t3,t4=kf.split(X)
#print(t1,t2,t3,t4)
test_index_total = []
for train_index, test_index in kf.split(X):
#   print("TRAIN:", train_index, "TEST:", test_index)
   test_index_total.extend(test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
   print("X_train:", X[train_index][:5])
   print("X_test: ", X[test_index][:5])
   print("y_train:", y[train_index][:5])
   print("y_test: ", y[test_index][:5])
   
test_index_total.sort()
print(test_index_total)
   