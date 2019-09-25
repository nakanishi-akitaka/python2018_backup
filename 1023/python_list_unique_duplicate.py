# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-list-unique-duplicate/
Created on Tue Oct 23 14:23:16 2018

@author: Akitaka
"""

l = [3, 3, 2, 1, 5, 1, 4, 2, 3]
l_unique = list(set(l))
print(l_unique)

print(dict.fromkeys(l))
l_unique_order = list(dict.fromkeys(l))
print(l_unique_order)

l_unique_order = sorted(set(l), key=l.index)
print(l_unique_order)

#%%
l_2d = [[0], [2], [2], [1], [0], [0]]
print(l_2d)

def get_unique_list(seq):
    seen = []
    return [x for x in seq if x not in seen and not seen.append(x)]

l_2d_unique = get_unique_list(l_2d)
print(l_2d_unique)

l_unique = get_unique_list(l)
print(l_unique)

#%%
l = [3, 3, 2, 1, 5, 1, 4, 2, 3]
l_duplicate = [x for x in set(l) if l.count(x) > 1]
print(l_duplicate)

l_duplicate_order = [x for x in dict.fromkeys(l) if l.count(x) > 1]
print(l_duplicate_order)

l_duplicate_order = sorted([x for x in set(l) if l.count(x) > 1], key=l.index)
print(l_duplicate_order)

#%%
l_2d = [[0], [2], [2], [1], [0], [0]]
print(l_2d)

def get_duplicate_list(seq):
    seen = []
    return [x for x in seq if not seen.append(x) and seen.count(x) == 2]

def get_duplicate_list_order(seq):
    seen = []
    return [x for x in seq if seq.count(x) > 1 and not seen.append(x) and seen.count(x) == 1]

l_2d_duplicate = get_duplicate_list(l_2d)
print(l_2d_duplicate)

l_2d_duplicate_order = get_duplicate_list_order(l_2d)
print(l_2d_duplicate_order)

l_duplicate = get_duplicate_list(l)
print(l_duplicate)

l_duplicate_order = get_duplicate_list_order(l)
print(l_duplicate_order)
