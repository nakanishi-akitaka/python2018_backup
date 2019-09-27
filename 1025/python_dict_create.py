# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-dict-create/
Created on Thu Oct 25 10:47:25 2018

@author: Akitaka
"""
d = {'k1':1, 'k2':2, 'k3':3}
print(d)

d = {'k1':1, 'k2':2, 'k3':3, 'k3':300}
print(d)

#%%
d = dict(k1=1, k2=2, k3=3)
print(d)

d = dict([('k1', 1), ('k2', 2), ('k3', 4)])
d = dict((['k1', 1], ['k2', 2], ['k3', 4]))
d = dict([{'k1', 1}, {'k2', 2}, {'k3', 4}])
print(d)

#%%
keys = ['k1', 'k2', 'k3']
values = [1, 2, 3]
d = dict(zip(keys, values))
print(d)

#%%
d_other = {'k10': 10, 'k100': 100}
print(d_other, type(d_other))
d = dict(d_other)
print(d, type(d))

print(d == d_other)
print(d is d_other)

#%%
l = ['Alice', 'Bob', 'Charlie']
d = {s: len(s) for s in l}
print(d)

#%%
keys = ['k1', 'k2', 'k3']
values = [1, 2, 3]
d = {k: v for k, v in zip(keys, values)}
print(d)

d = {k: v for k, v in zip(keys, values) if v % 2 == 1}
print(d)
