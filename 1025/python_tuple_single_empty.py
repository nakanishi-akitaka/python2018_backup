# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-tuple-single-empty/
Created on Thu Oct 25 10:38:47 2018

@author: Akitaka
"""
single_tuple = (0, )
print(single_tuple)
print(type(single_tuple))
print((0, 1, 2) + (3, ))

#%%
t = 0, 1, 2
print(t)
print(type(t))
t_ = 0,
print(t_)
print(type(t_))

#%%
empty_tuple = ()
print(empty_tuple)
print(type(empty_tuple))

#%%
def example(a, b):
    print(a, type(a))
    print(b, type(b))
example(0, 1)
example((0, 1), 2)
example(*(0, 1))
