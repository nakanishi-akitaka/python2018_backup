# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-list-tuple-convert/
Created on Wed Oct 24 10:13:58 2018

@author: Akitaka
"""

l = [0, 1, 2]
print(l)
print(type(l))

t = ('one', 'two', 'three')
print(t)
print(type(t))

r = range(10)
print(r)
print(type(r))

#%%
tl = list(t)
print(tl)
print(type(tl))

rl = list(r)
print(rl)
print(type(rl))

#%%
lt = tuple(l)
print(lt)
print(type(lt))

rt = tuple(r)
print(rt)
print(type(rt))

#%%
tl = list(t)
tl[0] = 'ONE'
t_new = tuple(tl)
print(t_new)
print(type(t_new))

#%%

t2 = t + ('four', 'five')
print(t)
print(t2)

t2 = t + ('four', )
print(t)
print(t2)

tl = list(t)
tl.insert(2, 'XXX')
t_new = tuple(tl)
print(t_new)
print(type(t_new))
