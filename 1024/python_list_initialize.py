# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-list-initialize/
Created on Wed Oct 24 10:05:06 2018

@author: Akitaka
"""

l_empty = []
print(l_empty)
print(len(l_empty))

l_empty.append(100)
l_empty.append(200)
print(l_empty)
l_empty.remove(100)
print(l_empty)

l = [0] * 10
print(l)
print(len(l))
print([0, 1, 2] * 3)

l_2d_ng=[[0] * 4] * 3
print(l_2d_ng)
l_2d_ng[0][0] = 5
print(l_2d_ng)
l_2d_ng[0].append(100)
print(l_2d_ng)

print(id(l_2d_ng[0]) == id(l_2d_ng[1]) == id(l_2d_ng[2]))

#%%
l_2d_ok = [[0] * 4 for i in range(3)]
print(l_2d_ok)

l_2d_ok[0][0] = 100
print(l_2d_ok)

print(id(l_2d_ok[0]) == id(l_2d_ok[1]) == id(l_2d_ok[2]))

#%%
l_2d_ok_2 = [[0] * 4 for i in [1] * 3]

print(l_2d_ok_2)

l_2d_ok_2[0][0] = 100
print(l_2d_ok_2)

print(id(l_2d_ok_2[0]) == id(l_2d_ok_2[1]) == id(l_2d_ok_2[2]))

#%%
t = (0,) * 5
print(t)
import array
a = array.array('i', [0] * 5)
print(a)


