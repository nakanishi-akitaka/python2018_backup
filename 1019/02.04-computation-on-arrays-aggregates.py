# -*- coding: utf-8 -*-
"""
https://jakevdp.github.io/PythonDataScienceHandbook/02.04-computation-on-arrays-aggregates.html
Created on Fri Oct 19 13:38:49 2018

@author: Akitaka
"""
# Summing the Values in an Array
import numpy as np

L = np.random.random(100)
print(sum(L))
print(np.sum(L))

big_array = np.random.rand(10000)

#%%
# Minimum and Maximum
print(min(big_array), max(big_array))
print(np.min(big_array), np.max(big_array))
print(big_array.min(), big_array.max(), big_array.sum())

#%%
## Multi dimensional aggregates
M = np.random.random((3, 4))
print(M)
print(M.sum())
print(M.min(axis=0))
print(M.min(axis=1))

#%%
## Other aggregation functions

#%%
# Example: What is the Average Height of US Presidents?
import pandas as pd
data = pd.read_csv('data/predident_heights.csv')
heights = np.array(data['height(cm)'])
print(heights)

print("Mean height:       ", heights.mean())
print("Standard deviation:", heights.std())
print("Minimum height:    ", heights.min())
print("Maximum height:    ", heights.max())

print("25th percentile:", np.percentile(heights, 25))
print("Median:         ", np.median(heights))
print("75th percentile:", np.percentile(heights, 75))

import matplotlib.pyplot as plt
import seaborn; seaborn.set()
plt.hist(heights)
plt.title('Height Distribution of US Presidents')
plt.xlabel('height (cm)')
plt.ylabel('number');








