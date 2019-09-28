# -*- coding: utf-8 -*-
"""
https://jakevdp.github.io/PythonDataScienceHandbook/04.00-introduction-to-matplotlib.html
Created on Fri Oct 26 13:13:54 2018

@author: Akitaka
"""

# import matplotlib as mpl
import matplotlib.pyplot as plt

## Setting Styles
# plt.style.use('classic')

## show() or No show()? How to Display Your Plots
### Plotting from a script
import numpy as np

x = np.linspace(0, 10, 100)

plt.xkcd()
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.show()

fig = plt.figure()
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.cos(x), '--');

#%%
## Saving Figures to File
fig.savefig('my_figure.png')


from IPython.display import Image
Image('my_figure.png')

#%%
# Two Interfaces for the Price of One
### MATLAB-style Interface
plt.figure()  # create a plot figure

# create the first of two panels and set current axis
plt.subplot(2, 1, 1) # (rows, columns, panel number)
plt.plot(x, np.sin(x))

# create the second panel and set current axis
plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x));



