# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-numpy-list/
Created on Fri Oct 19 11:11:54 2018

@author: Akitaka
"""

import numpy as np

l_1d = [0, 1, 2]

arr_1d = np.array(l_1d)

print(arr_1d)
print(arr_1d.dtype)

#%%
arr_1d_f = np.array(l_1d, dtype=float)

print(arr_1d_f)
print(arr_1d_f.dtype)

#%%
l_2d = [[0, 1, 2], [3, 4, 5]]

arr_2d = np.array(l_2d)

print(arr_2d)

#%%
l_2d_error = [[0, 1,2 ], [3, 4]]

arr_2d_error = np.array(l_2d_error)

print(arr_2d_error)
print(arr_2d_error.dtype)
print(arr_2d_error.shape)

#%%
arr_1d = np.arange(3)

print(arr_1d)

l_1d = arr_1d.tolist()

print(l_1d)

#%%
arr_2d = np.arange(6).reshape((2, 3))

print(arr_2d)

l_2d = arr_2d.tolist()

print(l_2d)

#%%
arr_3d = np.arange(24).reshape((2, 3, 4))

print(arr_3d)

l_3d = arr_3d.tolist()

print(l_3d)

print(l_3d[0])

print(l_3d[0][0])

print(l_3d[0][0][0])



