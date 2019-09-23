# -*- coding: utf-8 -*-
"""
https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html
Created on Fri Oct 19 14:09:33 2018

@author: Akitaka
"""
# Introducing Broadcasting
import numpy as np
a = np.array([0, 1, 2])
b = np.array([5, 5, 5])
print(a + b)
print(a + 5)

M = np.ones((3, 3))
print(M)
print(M + a)

a = np.arange(3)
b = np.arange(3)[:,np.newaxis]
print(a)
print(b)
print(a + b)

#%%
# Rules of Broadcasting
## Broadcasting example 1
M = np.ones((2, 3))
a = np.arange(3)
print(M)
print(a)
print(M + a)

## Broadcasting example 2
a = np.arange(3).reshape((3, 1))
b = np.arange(3)
print(a)
print(b)
print(a + b)

## Broadcasting example 3
M = np.ones((3, 2))
a = np.arange(3)
print(M)
print(a)
# print(M + a)

print(a[:, np.newaxis].shape)
print(M + a[:, np.newaxis])
print(np.logaddexp(M, a[:, np.newaxis]))

#%%
# Broadcasting in Practice
## Centering an array
X = np.random.random((10, 3))
print(X)

Xmean = X.mean(0)
print(Xmean)

X_centered = X - Xmean
print(X_centered.mean(0))

#%%
## Plotting a two-dimensional function
x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 50)[:, np.newaxis]
z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

import matplotlib.pyplot as plt
plt.imshow(z, origin='lower', extent=[0, 5, 0, 5], cmap='viridis')
plt.colorbar();



