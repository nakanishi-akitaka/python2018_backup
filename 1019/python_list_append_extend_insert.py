# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-list-append-extend-insert/
Created on Fri Oct 19 11:18:08 2018

@author: Akitaka
"""

l = list(range(3))
print(l)

l.append(100)
print(l)

l.append('new')
print(l)

l.append([3, 4, 5])
print(l)

#%%
l = list(range(3))
print(l)

l.extend([100, 101, 102])
print(l)

l.extend((-1, -2, -3))
print(l)

#%%
l.extend('new')
print(l)

l2 = l + [5, 6, 7]
print(l2)

l += [5, 6, 7]
print(l)

#%%
l = list(range(3))
print(l)

l.insert(0, 100)
print(l)

l.insert(-1, 200)
print(l)

l.insert(0, [-1, -2, -3])
print(l)

#%%
l = list(range(3))
print(l)

l[1:1] = [100, 200, 300]
print(l)

#%%
l = list(range(3))
print(l)

l[1:2] = [100, 200, 300]
print(l)



