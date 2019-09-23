# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-list-array-numpy-ndarray/
Created on Fri Oct 19 10:49:42 2018

@author: Akitaka
"""
l = ['apple', 100, 0.123]
print(l)

l_2d = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
print(l_2d)
print(l_2d[1][1])

#%%
l_num = [0, 10, 20, 30]
print(min(l_num))
print(max(l_num))
print(sum(l_num))
print(sum(l_num) / len(l_num))

#%%
l_str = ['apple', 'orange', 'banana']
for s in l_str:
    print(s)

#%% 
import array
arr_int = array.array('i', [0, 1, 2])
print(arr_int)

arr_float = array.array('f', [0.0, 0.1, 0.2])
print(arr_float)

#%%
print(arr_int[1])
print(sum(arr_int))

#%%

import numpy as np

arr = np.array([0, 1, 2])
print(arr)

arr_2d = np.array([[0, 1, 2], [3, 4, 5]])
print(arr_2d)

#%%
arr_2d_sqrt = np.sqrt(arr_2d)
print(arr_2d_sqrt)

arr_1 = np.array([[1, 2], [3, 4]])
arr_2 = np.array([[1, 2, 3], [4, 5, 6]])

arr_product = np.dot(arr_1, arr_2)
print(arr_product)








