# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-dict-in-values-items/
Created on Thu Oct 25 11:03:58 2018

@author: Akitaka
"""
d = {'key1': 'val1', 'key2': 'val2', 'key3': 'val3'}
print('key1' in d)
print('val1' in d)
print('key4' not in d)
print(d['key1'])
print(d.get('key4'))
print('val1' in d.values())
print('val4' not in d.values())

#%%
print(('key1', 'val1') in d.items())
print(('key1', 'val2') in d.items())
print(('key1', 'val2') not in d.items())
