# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-list-clear-pop-remove-del/
Created on Fri Oct 19 11:47:36 2018

@author: Akitaka
"""

l = list(range(10))
print(l)

l.clear()
print(l)

#%%
l = list(range(10))
print(l.pop(0))

print(l)

print(l.pop(3))

print(l)

#%%
print(l.pop(-2))

print(l)

print(l.pop())

print(l)

#%%
l = list('abcdefg')
print(l)

l.remove('d')
print(l)

l = [0, 1, 2, 1, 3]
l.remove(1)
print(l)

#%%
l = list(range(10))
print(l)

del l[0]
print(l)

del l[-1]
print(l)

del l[6]
print(l)


#%%
l = list(range(10))
print(l)

del l[2:5]
print(l)

l = list(range(10))
del l[:3]
print(l)

l = list(range(10))
del l[4:]
print(l)

l = list(range(10))
del l[-3:]
print(l)

l = list(range(10))
del l[:]
print(l)

#%%
l = list(range(10))
del l[2:8:2]
print(l)

l = list(range(10))
del l[::3]
print(l)

