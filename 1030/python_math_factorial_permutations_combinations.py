# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-math-factorial-permutations-combinations/
Created on Tue Oct 30 13:04:22 2018

@author: Akitaka
"""

import math

print(math.factorial(5))
# 120

print(math.factorial(0))
# 1

# print(math.factorial(1.5))
# ValueError: factorial() only accepts integral values

# print(math.factorial(-1))
# ValueError: factorial() not defined for negative values


#%%
def permutations_count(n, r):
    return math.factorial(n) // math.factorial(n - r)

print(permutations_count(4, 2))
# 12

print(permutations_count(4, 4))
# 24

#%%
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

p_list = list(itertools.permutations(l, 2))

print(p_list)
# [('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'a'), ('b', 'c'), ('b', 'd'), ('c', 'a'), ('c', 'b'), ('c', 'd'), ('d', 'a'), ('d', 'b'), ('d', 'c')]

print(len(p_list))
# 12

for v in itertools.permutations(l):
    print(v)
# ('a', 'b', 'c', 'd')
# ('a', 'b', 'd', 'c')
# ('a', 'c', 'b', 'd')
# ('a', 'c', 'd', 'b')
# ('a', 'd', 'b', 'c')
# ('a', 'd', 'c', 'b')
# ('b', 'a', 'c', 'd')
# ('b', 'a', 'd', 'c')
# ('b', 'c', 'a', 'd')
# ('b', 'c', 'd', 'a')
# ('b', 'd', 'a', 'c')
# ('b', 'd', 'c', 'a')
# ('c', 'a', 'b', 'd')
# ('c', 'a', 'd', 'b')
# ('c', 'b', 'a', 'd')
# ('c', 'b', 'd', 'a')
# ('c', 'd', 'a', 'b')
# ('c', 'd', 'b', 'a')
# ('d', 'a', 'b', 'c')
# ('d', 'a', 'c', 'b')
# ('d', 'b', 'a', 'c')
# ('d', 'b', 'c', 'a')
# ('d', 'c', 'a', 'b')
# ('d', 'c', 'b', 'a')

print(len(list(itertools.permutations(l))))
# 24

l = ['a', 'a']

for v in itertools.permutations(l, 2):
    print(v)
# ('a', 'a')
# ('a', 'a')

#%%
def combinations_count(n, r):
    return math.factorial(n) // (math.factorial(n - r) * math.factorial(r))

print(combinations_count(4, 2))
# 6

#%%
l = ['a', 'b', 'c', 'd']

c = itertools.combinations(l, 2)

print(type(c))
# <class 'itertools.combinations'>

for v in itertools.combinations(l, 2):
    print(v)
# ('a', 'b')
# ('a', 'c')
# ('a', 'd')
# ('b', 'c')
# ('b', 'd')
# ('c', 'd')

c_list = list(itertools.combinations(l, 2))

print(c_list)
# [('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')]

print(len(c_list))
# 6

#%%
def combinations_with_replacement_count(n, r):
    return combinations_count(n + r - 1, r)

print(combinations_with_replacement_count(4, 2))
# 10

#%%
h = itertools.combinations_with_replacement(l, 2)

print(type(h))
# <class 'itertools.combinations_with_replacement'>

for v in itertools.combinations_with_replacement(l, 2):
    print(v)
# ('a', 'a')
# ('a', 'b')
# ('a', 'c')
# ('a', 'd')
# ('b', 'b')
# ('b', 'c')
# ('b', 'd')
# ('c', 'c')
# ('c', 'd')
# ('d', 'd')

h_list = list(itertools.combinations_with_replacement(l, 2))

print(h_list)
# [('a', 'a'), ('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'b'), ('b', 'c'), ('b', 'd'), ('c', 'c'), ('c', 'd'), ('d', 'd')]

print(len(h_list))
# 10

#%%
s = 'arc'

for v in itertools.permutations(s):
    print(v)
# ('a', 'r', 'c')
# ('a', 'c', 'r')
# ('r', 'a', 'c')
# ('r', 'c', 'a')
# ('c', 'a', 'r')
# ('c', 'r', 'a')

anagram_list = [''.join(v) for v in itertools.permutations(s)]

print(anagram_list)
# ['arc', 'acr', 'rac', 'rca', 'car', 'cra']
