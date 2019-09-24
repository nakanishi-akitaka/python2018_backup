# -*- coding: utf-8 -*-
"""
https://jakevdp.github.io/PythonDataScienceHandbook/02.07-fancy-indexing.html
Created on Mon Oct 22 20:38:11 2018

@author: Akitaka
"""

# Exploring Fancy Indexing
import numpy as np
rand = np.random.RandomState(42)
x = rand.randint(100, size=10)
print(x)
print([x[3], x[7], x[2]])

ind = [3, 7, 4]
print(x[ind])

ind = np.array([[3, 7], [4, 5]])
print(x[ind])

X = np.arange(12).reshape((3, 4))
print(X)
row = np.array([0, 1, 2])
col = np.array([2, 1, 3])
print(X[row, col])

print(X[row[:, np.newaxis], col])
print(row[:, np.newaxis] * col)

#%%
# Combined Indexing

print(X)
print(X[2, [2, 0, 1]])
print(X[1:, [2, 0, 1]])

mask = np.array([1, 0, 1, 0], dtype=bool)
print(X[row[:, np.newaxis], mask])

#%%
# Example: Selecting Random Points

mean = [0, 0]
cov = [[1, 2], [2, 5]]
X = rand.multivariate_normal(mean, cov, 100)
print(X.shape)

import matplotlib.pyplot as plt
import seaborn; seaborn.set()
plt.scatter(X[:, 0], X[:,1])

#%%
indices = np.random.choice(X.shape[0], 20, replace=False)
print(indices)
selection = X[indices]
print(selection.shape)
plt.scatter(X[:, 0], X[:, 1], alpha=0.3)
plt.scatter(selection[:, 0], selection[:, 1], facecoloer='none', s=200);

#%%
# Modifying Values with Fancy Indexing
x = np.arange(10)
i = np.array([2, 1, 8, 4])
x[i] = 99
print(x)

x[i] -= 10
print(x)

x = np.zeros(10)
x[[0, 0]] = [4, 6]
print(x)

i = [2, 3, 3, 4, 4, 4]
x[i] += 1
print(x)

x = np.zeros(10)
np.add.at(x, i, 1)
print(x)

#%%
# Example: Binning Data
np.random.seed(42)
x = np.random.randn(100)

bins = np.linspace(-5, 5, 20)
counts = np.zeros_like(bins)

i = np.searchsorted(bins, x)
np.add.at(counts, i, 1)

plt.plot(bins, counts, linestyle='steps');
plt.hist(x, bins, histtype='step');

