# -*- coding: utf-8 -*-
"""
https://jakevdp.github.io/PythonDataScienceHandbook/02.06-boolean-arrays-and-masks.html
Created on Fri Oct 19 14:23:28 2018

@author: Akitaka
"""
import numpy as np
import pandas as pdf

rainfall = pd.read_csv('Seattle2014.csv')['PRCP'].values
inches = rainfall / 254.0
inches.shape

import matplotlib.pyplot asplt
import seaborn; seaborn.set()
plt.hist(inches, 40);

## Digging into the data

#%%
# Comparison Operators as ufuncs
x = np.array([1, 2, 3, 4, 5])
print(x < 3)
print(x > 3)
print(x <= 3)
print(x >= 3)
print(x != 3)
print(x == 3)
print((2 * x) == (x ** 2))
rng = np.random.RandomState(0)
x = rng.randint(10, size=(3, 4))
print(x)
print(x < 6)

#%%
# Working with Boolean Arrays
print(x)

## Counting entries
print(np.count_nonzero(x < 6))
print(np.sum(x < 6))
print(np.sum(x < 6, axis=1))

#%%
print(np.any(x > 0))
print(np.any(x < 0))
print(np.all(x < 10))
print(np.all(x == 0))
print(np.all(x < 8, axis=1))


#%%
## Boolean operators

np.sum((inches > 0.5) & (inches < 1))
inches > (0.5 & inches) < 1
np.sum(~((inches <= 0.5) | (inches >= 1)))

print("Number days without rain:", np.sum(inches == 0))
print("Number days with rain:", np.sum(inches != 0))
print("Dayes with more than 0.5 inches:", np.sum(inches > 0.5))
print("Rainy days with < 0.2 inches:", np.sum((inches > 0) & (inches < 0.2)))

#%%
# Boolean Arrays as Masks
print(x)
print(x < 5)
print(x[x < 5])

# construct a mask of all rainy days
rainy = (inches > 0)

# construct a mask of all summer days (June 21st is the 172nd day)
days = np.arange(365)
summer = (days > 172) & (days < 262)

print("Median precip on rainy days in 2014 (inches):   ",
      np.median(inches[rainy]))
print("Median precip on summer days in 2014 (inches):  ",
      np.median(inches[summer]))
print("Maximum precip on summer days in 2014 (inches): ",
      np.max(inches[summer]))
print("Median precip on non-summer rainy days (inches):",
      np.median(inches[rainy & ~summer]))

#%%
# Aside: Using the Keywords and/or Versus the Operators &/|
print(bool(42), bool(0))
print(bool(42 and 0))
print(bool(42 or 0))

print(bin(42))
print(bin(59))
print(bin(42 & 59))
print(bin(42 | 59))

#%%
A = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
B = np.array([1, 1, 1, 0, 1, 1], dtype=bool)
print(A | B)

x = np.arange(10)
print((x > 4) & (x < 8))
print((x > 4) and (x < 8))






