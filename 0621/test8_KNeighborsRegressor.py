# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 14:20:41 2018

@author: Akitaka
http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
"""

X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
X_test = [[1.5]]
from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X, y) 
print(neigh.predict(X_test))

samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(samples) 
print("Q. who’s the closest point to [1,1,1] ?")
print("A. distance, index of point, datatype = ")
print(neigh.kneighbors([[1., 1., 1.]])) 
X = [[0., 1., 0.], [1., 0., 1.]]
print("Q. who’s the closest points to", X, "? (multi version)")
print("A. index list of points = ")
print(neigh.kneighbors(X, return_distance=False)) 

X = [[0], [3], [1]]
from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(n_neighbors=2)
neigh.fit(X) 
A = neigh.kneighbors_graph(X)
print("Computes the (weighted) graph of k-Neighbors for points in X")
print("k = ", 2-1, " X = ", X)
print("(point 1, 2) distance")
print(A)
print(A.toarray())
