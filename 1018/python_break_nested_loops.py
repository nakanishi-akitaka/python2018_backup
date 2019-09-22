# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-break-nested-loops/
Created on Thu Oct 18 13:04:55 2018

@author: Akitaka
"""

l1 = [1, 2, 3]
l2 = [10, 20, 30]
for i in l1:
    for j in l2:
        print(i, j)

print()
for i in l1:
    for j in l2:
        print(i, j)
        if i == 2 and j == 20 :
            print('BREAK')
            break

print()
for i in l1:
    for j in l2:
        print(i, j)
        if i == 2 and j == 20 :
            print('BREAK')
            break
    else:
        continue
    break

print()
for i in l1:
    print('Start outer loop')
    for j in l2:
        print('--', i, j)
        if i == 2 and j == 20 :
            print('-- BREAK inner loop')
            break
    else:
        print('-- Finish inner loop without BREAK')
        continue
    print('BREAK outer loop')
    break

l1 = [1, 2, 3]
l2 = [10, 20, 30]
l3 = [100, 200, 300]
for i in l1:
    for j in l2:
        for k in l3:
            print(i, j, k)
            if i == 2 and j == 20 and k == 200:
                print('BREAK')
                break
        else:
            continue
        break
    else:
        continue
    break

l1 = [1, 2, 3]
l2 = [10, 20, 30]

flag = False
for i in l1:
    for j in l2:
        print(i, j)
        if i == 2 and j == 20 :
            flag = True
            print('BREAK')
            break
    if flag:
        break


l1 = [1, 2, 3]
l2 = [10, 20, 30]
l3 = [100, 200, 300]
flag = False
for i in l1:
    for j in l2:
        for k in l3:
            print(i, j, k)
            if i == 2 and j == 20 and k == 200:
                flag = True
                print('BREAK')
                break
        if flag:
            break
    if flag:
        break

import itertools
l1 = [1, 2, 3]
l2 = [10, 20, 30]
for i, j in itertools.product(l1, l2):
    print(i, j)

print()
for i, j in itertools.product(l1, l2):
    print(i, j)
    if i == 2 and j == 20:
        print('BREAK')
        break

print()
l1 = [1, 2, 3]
l2 = [10, 20, 30]
l3 = [100, 200, 300]
for i, j, k in itertools.product(l1, l2, l3):
    print(i, j, k)
    if i == 2 and j == 20 and k == 200:
        print('BREAK')
        break

#%%
import itertools
from time                    import time
start = time()
n = 100
l1 = range(n)
l2 = range(n)
l3 = range(n)
x = n - 1
for i in l1:
    for j in l2:
        for k in l3:
            if i == x and j == x and k == x:
                break
        else:
            continue
        break
    else:
        continue
    break
print('{:.2f} seconds '.format(time() - start))

start = time()
flag = False
for i in l1:
    for j in l2:
        for k in l3:
            if i == x and j == x and k == x:
                flag = True
                break
        if flag:
            break
    if flag:
        break
print('{:.2f} seconds '.format(time() - start))


start = time()
for i, j, k in itertools.product(l1, l2, l3):
    if i == x and j == x and k == x:
        break
print('{:.2f} seconds '.format(time() - start))
