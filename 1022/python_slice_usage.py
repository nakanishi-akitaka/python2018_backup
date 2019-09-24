# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-slice-usage/
Created on Mon Oct 22 15:53:20 2018

@author: Akitaka
"""

l = [0, 10, 20, 30, 40, 50, 60]
print(l)
print(l[2:5])
print(l[:3])
print(l[3:])
print(l[:])
print(l[2:10])

print(l[5:2])
print(l[2:2])
print(l[10:20])

print(l[::2])
print(l[1::2])
print(l[::3])
print(l[2:5:2])

#%%
print(l[3:-1])
print(l[-2:])
print(l[-5:-2])

print(l[5:2:-1])
print(l[2:5:-1])

print(l[-2:-5:-1])
print(l[-2:2:-1])
print(l[5:2:-2])

print(l[::-1])

#%%

sl = slice(2, 5, 2)
print(sl)

print(type(sl))

print(l[sl])

sl = slice(2, 5)
print(sl)
print(l[sl])

sl = slice(2)
print(sl)
print(l[sl])


sl = slice(None)
print(sl)
print(l[sl])

#%%
l = [0, 10, 20, 30, 40, 50, 60]
print(l)

l[2:5] = [200, 300, 400]
print(l)

l[2:5] = [-2, -3]
print(l)

l[2:4] = [2000, 3000, 4000, 5000]
print(l)

l[2:6] = [20000]
print(l)

l[1:4] = []
print(l)

l[20:60] = [-1, -2, -3]
print(l)

l[2:2] = [-100]
print(l)

print(l[:5:2])
l[:5:2] = [100, 200, 300]
print(l)

#%%
l_2d = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
#import numpy as np
#l_2d = np.arange(12).reshape((4,3))
print(l_2d)
print(l_2d[1:3])
print([l[:2] for l in l_2d[1:3]])

#%%
l_2d_t = [list(x) for x in zip(*l_2d)]
print(l_2d_t)
print(l_2d_t[1])

#%%

l = [0, 10, 20, 30, 40, 50, 60]
print(l)
l_slice = l[2:5]
print(l_slice)

l_slice[1] = 300
print(l_slice)

print(l)

#%%
l_2d = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
print(l_2d)
l_2d_slice = l_2d[1:3]
print(l_2d_slice)
l_2d_slice[0][1] = 400
print(l_2d_slice)
print(l_2d)

#%%
import copy
l_2d = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
print(l_2d)
l_2d_slice_deepcopy = copy.deepcopy(l_2d[1:3])
print(l_2d_slice_deepcopy)
l_2d_slice_deepcopy[0][1] = 400
print(l_2d_slice_deepcopy)
print(l_2d)

#%%
s = 'abcdefg'
print(s)
print(s[2:5])
print(s[2:5])
print(s[::-1])

t = (0, 10, 20, 30, 40, 50, 60)
print(t)
print(t[2:5])






