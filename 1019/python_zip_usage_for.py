# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-zip-usage-for/
Created on Fri Oct 19 12:41:06 2018

@author: Akitaka
"""
names = ['Alice', 'Bob', 'Charlie']
ages = [24, 50, 18]

for name, age in zip(names, ages):
    print(name, age)


#%%
points = [100, 85, 90]

for name, age, point in zip(names, ages, points):
    print(name, age, point)

#%%
names = ['Alice', 'Bob', 'Charlie', 'Dave']
ages = [24, 50, 18]
for name, age in zip(names, ages):
    print(name, age)

#%%
from itertools import zip_longest
names = ['Alice', 'Bob', 'Charlie', 'Dave']
ages = [24, 50, 18]
for name, age in zip_longest(names, ages):
    print(name, age)

#%%
for name, age in zip_longest(names, ages, fillvalue=20):
    print(name, age)

#%%
names = ['Alice', 'Bob', 'Charlie']
ages = (24, 50, 18)

z = zip(names, ages)
print(z)
print(type(z))

#%%
l = list(zip(names, ages))
print(l)
print(type(l))
print(type(l[0]))








