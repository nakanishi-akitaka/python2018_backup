# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-for-usage/
Created on Thu Oct 18 12:26:21 2018

@author: Akitaka
"""

l = ['Alice', 'Bob', 'Charlie']

for name in l:
    print(name)

print()
for name in l:
    if name == 'Bob':
        print('!!BREAK!!')
        break
    print(name)

print()
for name in l:
    if name == 'Bob':
        print('!!BREAK!!')
        continue
    print(name)

print()
for name in l:
    print(name)
else:
    print('!!FINISH!!')

print()
for name in l:
    if name == 'Bob':
        print('!!BREAK!!')
        break
    print(name)
else:
    print('!!FINISH!!')

print()
for name in l:
    if name == 'Bob':
        print('!!BREAK!!')
        continue
    print(name)
else:
    print('!!FINISH!!')

#%%
for i in range(3):
    print(i)

print(range(3))
print(type(range(3)))

print(list(range(3)))
print(list(range(6)))
print(list(range(10, 13)))
print(list(range(0, 10, 3)))
print(list(range(10, 0, -3)))

for i in range(10, 0, -3):
    print(i)


#%%

l = ['Alice', 'Bob', 'Charlie']


for name in l:
    print(name)

for i, name in enumerate(l):
    print(i, name)

for i, name in enumerate(l, 1):
    print(i, name)

for i, name in enumerate(l, 42):
    print(i, name)

step = 3
for i, name in enumerate(l):
    print(i * step, name)

#%%
names = ['Alice', 'Bob', 'Charlie']
ages = [24, 50, 18]

for name, age in zip(names, ages):
    print(name, age)

points = [100, 85, 90]
for name, age, point in zip(names, ages, points):
    print(name, age, point)

for i, (name, age) in enumerate(zip(names, ages)):
    print(i, name, age)

l = ['Alice', 'Bob', 'Charlie']
for name in reversed(l):
    print(name)

for i in reversed(range(3)):
    print(i)

for i in range(2, -1, -1):
    print(i)

for i, name in reversed(list(enumerate(l))):
    print(i, name)

for i, name in enumerate(reversed(l)):
    print(i, name)

l2 = [24, 50, 18]
for name, age in reversed(list(zip(l, l2))):
    print(name, age)

#%%
l1 = [1, 2, 3]
l2 = [10, 20, 30]
for i in l1:
    for j in l2:
        print(i, j)

print()
import itertools
for i, j in itertools.product(l1, l2):
    print(i, j)

print()
d = {'key1': 1, 'key2': 2, 'key3': 3}
for k in d:
    print(k)

for v in d.values():
    print(v)

for k, v in d.items():
    print(k, v)

#%%
squares = [i**2 for i in range(5)]
print(squares)

squares = []
for i in range(5):
    squares.append(i**2)
print(squares)









