# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-dict-clear-pop-popitem-del/
Created on Thu Oct 25 11:00:03 2018

@author: Akitaka
"""

d = {'k1': 1, 'k2': 2, 'k3': 3}
d.clear()
print(d)

#%%
d = {'k1': 1, 'k2': 2, 'k3': 3}
removed_value = d.pop('k1')
print(d)
print(removed_value)

d = {'k1': 1, 'k2': 2, 'k3': 3}
removed_value = d.pop('k4', None)
print(d)
print(removed_value)

#%%
d = {'k1': 1, 'k2': 2}
k, v = d.popitem()
print(k)
print(v)
print(d)

k, v = d.popitem()
print(k)
print(v)
print(d)

#%%
d = {'k1': 1, 'k2': 2, 'k3': 3}
del d['k2']
print(d)

d = {'k1': 1, 'k2': 2, 'k3': 3}
del d['k1'], d['k3']
print(d)
