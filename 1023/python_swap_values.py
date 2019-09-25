# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-swap-values/
Created on Tue Oct 23 14:20:47 2018

@author: Akitaka
"""

a = 1
b = 2
a, b = b, a
print(a, b)

a, b = 100, 200
print(a, b)

a, b, c, d = 0, 1, 2, 3
a, b, c, d = c, d, a, b
print(a, b, c, d)

l = [0, 1, 2, 3, 4]
l[0], l[3] = l[3], l[0]
print(l)
print(sorted(l))
print(sorted(l, reverse=True))

