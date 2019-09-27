# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-tuple-list-unpack/
Created on Thu Oct 25 10:42:10 2018

@author: Akitaka
"""
t = (0, 1, 2)
a, b, c = t
print(a, b, c)
l = [0, 1, 2]
a, b, c = l
print(a, b, c)

a, b = 0, 1
print(a, b)

#%%
t = (0, 1, (2, 3, 4))
a, b, c = t
print(a, b, c)
print(type(c))

a, b, (c, d, e) = t
print(a, b, c, d, e)

#%%
t = (0, 1, 2)
a, b, _ = t
print(a, b, _)

#%%
t = (0, 1, 2, 3, 4)
a, b, *c = t
print(a, b, c, type(c))

a, *b, c = t
print(a, b, c, type(b))

*a, b, c = t
print(a, b, c, type(a))

a, b, *_ = t
print(a, b, _)

a, b = t[0], t[1]
print(a, b)

t = (0, 1, 2)
a, b, *c = t
print(a, b, c, type(c))

a, b, c, *d = t
print(a, b, c, d, type(d))
