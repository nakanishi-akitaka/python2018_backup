# -*- coding: utf-8 -*-
"""
https://jakevdp.github.io/PythonDataScienceHandbook/02.08-sorting.html
Created on Mon Oct 22 20:59:17 2018

@author: Akitaka
"""

import numpy as np
def selection_sort(x):
    for i in range(len(x)):
        swap = i + np.argmin(x[i:])
        (x[i], x[swap]) = (x[swap], x[i])
    return x

x = np.array([2, 1, 4, 3, 5])
print(selection_sort(x))

def bogosort(x):
    while np.any(x[:-1] > x[1:]):
        np.random.shuffle(x)
    return x

print(bogosort(x))

#%%
# Fast Sorting in NumPy: np.sort and np.argsort
x = np.array([2, 1, 4, 3, 5])
print(np.sort(x))

x = np.array([2, 1, 4, 3, 5])
x.sort()
print(x)

x = np.array([2, 1, 4, 3, 5])
i = np.argsort(x)
print(i)
print(x[i])

#%%
# Sorting along rows or columns
rand = np.random.RandomState(42)
X = rand.randint(0, 10, (4, 6))
print(X)
print(np.sort(X, axis=0))
print(np.sort(X, axis=1))

#%%
# Partial Sorts: Partitioning
x = np.array([7, 2, 3, 1, 6, 5, 4])
print(np.partition(x, 3))
print(np.partition(X, 2, axis=1))

#%%
# Example: k-Nearest Neighbors
X = rand.rand(10, 2)
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
plt.scatter(X[:, 0], X[:, 1], s=100);

#%%
dist_sq = np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1)
differences = X[:, np.newaxis, :] - X[np.newaxis, :, :]
print(differences.shape)

sq_differences = differences ** 2
print(sq_differences.shape)

dis_sq = sq_differences.sum(-1)
print(dist_sq.shape)

print(dist_sq.diagonal())

nearest = np.argsort(dist_sq, axis=1)
print(nearest)

K = 2
nearest_partition = np.argpartition(dist_sq, K + 1, axis=1)

plt.scatter(X[:, 0], X[:, 1], s=100)
K = 2
for i in range(X.shape[0]):
    for j in nearest_partition[i, :K+1]:
        plt.plot(*zip(X[j], X[i]), color='black')

