# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-dict-keys-values-items/
Created on Thu Oct 25 11:14:16 2018

@author: Akitaka
"""
d = {'key1': 1, 'key2': 2, 'key3': 3}

for k in d:
    print(k)

#%%
for k in d.keys():
    print(k)

keys = d.keys()
print(keys, type(keys))

k_list = list(d.keys())
print(k_list, type(k_list))

#%%
for v in d.values():
    print(v)

values = d.values()
print(values, type(values))
v_list = list(d.values())
print(v_list, type(v_list))

#%%
for k, v in d.items():
    print(k, v)

for t in d.items():
    print(t, type(t))
    print(t[0], t[1])
    print('---')

#%%
items = d.items()
print(items, type(items))

i_list = list(d.items())
print(i_list, type(i_list))

print(i_list[0], type(i_list[0]))
