# -*- coding: utf-8 -*-
"""
https://jakevdp.github.io/PythonDataScienceHandbook/04.04-density-and-contour-plots.html
Created on Fri Oct 26 13:48:35 2018

@author: Akitaka
"""
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np

# Visualizing a Three-Dimensional Function
def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)
plt.xkcd()
plt.contour(X, Y, Z, colors='black');
#%%
plt.xkcd()
plt.contour(X, Y, Z, 20, cmap='RdGy');

#%%
plt.contourf(X, Y, Z, 20, cmap='RdGy')
plt.colorbar();

#%%
plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
           cmap='RdGy')
plt.colorbar()
plt.axis(aspect='image');

#%%
contours = plt.contour(X, Y, Z, 3, colors='black')
plt.clabel(contours, inline=True, fontsize=8)

plt.imshow(Z, extent=[0, 5, 0, 5], origin='lower',
           cmap='RdGy', alpha=0.5)
plt.colorbar();