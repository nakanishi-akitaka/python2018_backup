# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-list-len/
Created on Fri Oct 19 12:32:09 2018

@author: Akitaka
"""
l = [0, 1, 2, 3]
print(len(l))

l_length = len(l)
print(l_length)
print(type(l_length))

#%%
l_2d = [[0, 1, 2], [3, 4, 5]]
print(len(l_2d))
print([len(v) for v in l_2d])

print(sum(len(v) for v in l_2d))

#%%
import numpy as np
l_2d = [[0, 1, 2], [3, 4, 5]]
arr_2d = np.array(l_2d)
print(arr_2d)
print(arr_2d.size)
print(arr_2d.shape)


#%%
l_multi = [[0, 1, 2, [10, 20, 30]], [3, 4, 5], 100]
print(len(l_multi))

#%%
arr_multi = np.array(l_multi)
print(arr_multi)
print(arr_multi.size)
print(arr_multi.shape)

#%%
def my_len(l):
    count = 0
    if isinstance(l, list):
        for v in l:
            count += my_len(v)
        return count
    else:
        return 1

l_multi = [[0, 1, 2, [10, 20, 30]], [3, 4, 5], 100]
print(my_len(l_multi))

l_2d = [[0, 1, 2], [3, 4, 5]]
print(my_len(l_2d))

l = [0, 1, 2, 3]
print(my_len(l))



















