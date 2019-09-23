# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-enumerate-start/
Created on Fri Oct 19 12:40:51 2018

@author: Akitaka
"""

l = ['Alice', 'Bob', 'Charlie']
for name in l:
    print(name)

for i, name in enumerate(l):
    print(i, name)

#%%
for i, name in enumerate(l, 1):
    print(i, name)

for i, name in enumerate(l, 42):
    print(i, name)

for i, name in enumerate(l, 1):
    print('{:03}_{}'.format(i, name))


