# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-list-index/
Created on Fri Oct 19 12:40:31 2018

@author: Akitaka
"""

l = list('abcde')
print(l)
print(l.index('a'))
print(l.index('c'))

def my_index(l, x, default=False):
    if x in l:
        return l.index(x)
    else:
        return default

print(my_index(l, 'd'))
print(my_index(l, 'x'))
print(my_index(l, 'x', -1))

#%%
l_dup = list('abcba')
print(l_dup)
print(l_dup.index('a'))
print(l_dup.index('b'))

#%%
print([i for i, x in enumerate(l_dup) if x == 'a'])
print([i for i, x in enumerate(l_dup) if x == 'b'])

print([i for i, x in enumerate(l_dup) if x == 'c'])

print([i for i, x in enumerate(l_dup) if x == 'x'])

#%%
def my_index_multi(l, x):
    return [i for i, _x in enumerate(l) if _x == x]

print(my_index_multi(l_dup, 'a'))
print(my_index_multi(l_dup, 'c'))
print(my_index_multi(l_dup, 'x'))

#%%
t = tuple('abcde')
print(t)
print(t.index('a'))

print(my_index(t, 'c'))
print(my_index(t, 'x'))

t_dup = tuple('abcba')
print(t_dup)
print(my_index_multi(t_dup, 'a'))














