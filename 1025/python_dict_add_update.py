# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-dict-add-update/
Created on Thu Oct 25 10:56:23 2018

@author: Akitaka
"""

d = {'k1': 1, 'k2': 2}
d['k3'] = 3
print(d)
d['k1'] = 100
print(d)

#%%
d1 = {'k1': 1, 'k2': 2}
d2 = {'k1': 100, 'k3': 3, 'k4': 4}
d1.update(d2)
print(d1)

d = {'k1': 1, 'k2': 2}
d.update(k1=100, k3=3, k4=4)
print(d)

d = {'k1': 1, 'k2': 2}
d.update([('k1', 100), ('k3', 3), ('k4', 4)])
print(d)

d = {'k1': 1, 'k2': 2}
keys = ['k1', 'k3', 'k4']
values = [100, 3, 4]
d.update(zip(keys, values))
print(d)


