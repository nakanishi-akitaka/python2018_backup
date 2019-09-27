# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-dict-value-max-min/
Created on Thu Oct 25 11:32:06 2018

@author: Akitaka
"""
d = {'a': 100, 'b': 20, 'c': 50, 'd': 100, 'e': 80}
print(max(d))
print(min(d))
print(max(d.values()))
print(min(d.values()))
print(max(d, key=d.get))
print(min(d, key=d.get))

#%%
print(max(d.items(), key=lambda x: x[1]))
print(type(max(d.items(), key=lambda x: x[1])))

max_k, max_v = max(d.items(), key=lambda x: x[1])
print(max_k, max_v)
min_kv = min(d.items(), key=lambda x: x[1])
print(min_kv)

#%%
max_kv_list = [kv for kv in d.items() if kv[1] == max(d.values())]
print(max_kv_list)

max_k_list = [kv[0] for kv in d.items() if kv[1] == max(d.values())]
print(max_k_list)

min_kv_list = [kv for kv in d.items() if kv[1] == min(d.values())]
print(min_kv_list)

