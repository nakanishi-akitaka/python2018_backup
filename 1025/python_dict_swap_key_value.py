# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-dict-swap-key-value/
Created on Thu Oct 25 11:28:59 2018

@author: Akitaka
"""
d = {'key1': 'val1', 'key2': 'val2', 'key3': 'val3'}
d_swap = {v: k for k, v in d.items()}
print(d_swap)

def get_swap_dict(d):
    return {v: k for k, v in d.items()}

d_swap = get_swap_dict(d)
print(d_swap)

#%%
d_duplicate = {'key1': 'val1', 'key2': 'val1', 'key3': 'val3'}
d_duplicate_swap = get_swap_dict(d_duplicate)
print(d_duplicate_swap)



