# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:41:04 2018

@author: Akitaka
"""
names = ['Alice', 'Bob', 'Charlie']
ages = [24, 50, 18]

for i, (name, age) in enumerate(zip(names, ages)):
    print(i, name, age)

print()
for i, t in enumerate(zip(names, ages)):
    print(i, t)

print()
for i, t in enumerate(zip(names, ages)):
    print(i, t[0], t[1])