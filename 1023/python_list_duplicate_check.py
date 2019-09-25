# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-list-duplicate-check/
Created on Tue Oct 23 14:44:34 2018

@author: Akitaka
"""

def is_unique(seq):
    return len(seq) == len(set(seq))

l = [0, 'two', 1, 'two', 0]
print(is_unique(l))

l = [0, 'one', 2]
print(is_unique(l))

#%%
def is_unique2(seq):
    seen = []
    unique_list = [x for x in seq if x not in seen and not seen.append(x)]
    return len(seq) == len(unique_list)

l_2d = [[0, 1], [1, 1], [0, 1], [1, 0]]
print(is_unique2(l_2d))

l_2d = [[0, 1], [1, 1], [1, 0]]
print(is_unique2(l_2d))

#%%
l = [0, 'two', 1, 'two', 0]
print(is_unique2(l))

l = [0, 'one', 2]
print(is_unique2(l))


l = [[0, 1, 2], 'string', 100, [0, 1, 2]]
print(is_unique2(l))

l = [[0, 1, 2], 'string', 100]
print(is_unique2(l))
