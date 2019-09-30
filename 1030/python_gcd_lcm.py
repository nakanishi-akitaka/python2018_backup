# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-gcd-lcm/
Created on Tue Oct 30 11:55:16 2018

@author: Akitaka
"""

import math

a = 6
b = 4

print(math.gcd(a, b))
# 2

def lcm(x, y):
    return (x * y) // math.gcd(x, y)

print(lcm(a, b))
# 12


#%%
import math
from functools import reduce

def gcd(*numbers):
    return reduce(math.gcd, numbers)

def gcd_list(numbers):
    return reduce(math.gcd, numbers)

print(gcd(27, 18, 9))
# 9

print(gcd(27, 18, 9, 3))
# 3

print(gcd([27, 18, 9, 3]))
# [27, 18, 9, 3]

print(gcd(*[27, 18, 9, 3]))
# 3

print(gcd_list([27, 18, 9, 3]))
# 3

# print(gcd_list(27, 18, 9, 3))
# TypeError: gcd_list() takes 1 positional argument but 4 were given

#%%
def lcm_base(x, y):
    return (x * y) // math.gcd(x, y)

def lcm(*numbers):
    return reduce(lcm_base, numbers, 1)

def lcm_list(numbers):
    return reduce(lcm_base, numbers, 1)

print(lcm(27, 18, 9, 3))
# 54

print(lcm(27, 9, 3))
# 27

print(lcm(*[27, 9, 3]))
# 27

print(lcm_list([27, 9, 3]))
# 27



