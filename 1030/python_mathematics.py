# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-mathematics/
Created on Tue Oct 30 11:47:29 2018

@author: Akitaka
"""
print(10 + 3)
# 13

print(10 - 3)
# 7

print(10 * 3)
# 30

print(10 / 3)
# 3.3333333333333335

print(10 // 3)
# 3

print(10 % 3)
# 1

print(10 ** 3)
# 1000

#%%
from fractions import Fraction

print(Fraction(1, 3))
# 1/3

#%%
import math

a = 6
b = 4

print(math.gcd(a, b))
# 2

#%%
import math

print(math.pi)
# 3.141592653589793

print(math.degrees(math.pi))
# 180.0

print(math.radians(180))
# 3.141592653589793

sin30 = math.sin(math.radians(30))
print(sin30)
# 0.49999999999999994

#%%
print(math.log(25, 5))
# 2.0

print(math.log(math.e))
# 1.0

print(math.log10(100000))
# 5.0

print(math.log2(1024))
# 10.0

#%%
import math

print(math.factorial(5))
# 120

print(math.factorial(0))
# 1

import itertools

l = ['a', 'b', 'c', 'd']

p = itertools.permutations(l, 2)

print(type(p))
# <class 'itertools.permutations'>

for v in itertools.permutations(l, 2):
    print(v)
# ('a', 'b')
# ('a', 'c')
# ('a', 'd')
# ('b', 'a')
# ('b', 'c')
# ('b', 'd')
# ('c', 'a')
# ('c', 'b')
# ('c', 'd')
# ('d', 'a')
# ('d', 'b')
# ('d', 'c')

#%%
s = {1, 2, 2, 3, 1, 4}

print(s)
print(type(s))
# {1, 2, 3, 4}
# <class 'set'

s1 = {0, 1, 2}
s2 = {1, 2, 3}
s3 = {2, 3, 4}

s_union = s1 | s2
print(s_union)
# {0, 1, 2, 3}

#%%
c1 = 3 + 4j
c2 = 2 - 1j

print(c1 + c2)
# (5+3j)

print(c1 - c2)
# (1+5j)

print(c1 * c2)
# (10+5j)

print(c1 / c2)
# (0.4+2.2j)

#%%
import sympy

x = sympy.Symbol('x')
y = sympy.Symbol('y')

print(type(x))
# <class 'sympy.core.symbol.Symbol'>

print(sympy.factor(x**3 - x**2 - 3 * x + 3))
# (x - 1)*(x**2 - 3)

print(sympy.factor(x * y + x + y + 1))
# (x + 1)*(y + 1)

#%%
import numpy as np
arr1 = np.arange(4).reshape((2, 2))

print(arr1)
# [[0 1]
#  [2 3]]

arr2 = np.arange(6).reshape((2, 3))

print(arr2)
# [[0 1 2]
#  [3 4 5]]

arr_mul_matrix = np.dot(arr1, arr2)

print(arr_mul_matrix)
# [[ 3  4  5]
#  [ 9 14 19]]






