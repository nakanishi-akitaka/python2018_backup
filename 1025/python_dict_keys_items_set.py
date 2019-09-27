# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-dict-keys-items-set/
Created on Thu Oct 25 11:53:10 2018

@author: Akitaka
"""
d1 = {'a': 1, 'b': 2, 'c': 3}
d2 = {'b': 2, 'c': 4, 'd': 5}

print(list(d1.keys()))
print(type(d1.keys()))
print(list(d1.items()))
print(type(d1.items()))

#%%
intersection_keys = d1.keys() & d2.keys()
print(intersection_keys)
print(type(intersection_keys))

intersection_items = d1.items() & d2.items()
print(intersection_items)

intersection_dict = dict(d1.items() & d2.items())
print(intersection_dict)
print(type(intersection_dict))

#%%
union_keys = d1.keys() | d2.keys()
print(union_keys)

union_items = d1.items() | d2.items()
print(union_items)

union_dict = dict(d1.items() | d2.items())
print(union_dict)

#%%
symmetric_difference_keys = d1.keys() ^ d2.keys()
print(symmetric_difference_keys)

symmetric_difference_items = d1.items() ^ d2.items()
print(symmetric_difference_items)

symmetric_difference_dict = dict(d1.items() ^ d2.items())
print(symmetric_difference_dict)

#%%
difference_keys = d1.keys() - d2.keys()
print(difference_keys)

difference_items = d1.items() - d2.items()
print(difference_items)

difference_dict = dict(d1.items() - d2.items())
print(difference_dict)



