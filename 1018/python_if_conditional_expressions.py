# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-if-conditional-expressions/
Created on Thu Oct 18 13:39:08 2018

@author: Akitaka
"""

a = 1
result = 'even' if a % 2 == 0 else 'odd'
print(result)

a = 2
result = 'even' if a % 2 == 0 else 'odd'
print(result)

a = -2
result = 'negative and even' if a < 0 and a % 2 == 0 else 'positive or odd'
print(result)

a = -1
result = 'negative and even' if a < 0 and a % 2 == 0 else 'positive or odd'
print(result)

a = 2
result = 'negative' if a < 0 else 'positive' if a > 0 else 'zero'
print(result)

a = -2
result = 'negative' if a < 0 else 'positive' if a > 0 else 'zero'
print(result)

a = 0
result = 'negative' if a < 0 else 'positive' if a > 0 else 'zero'
print(result)

#%%
l = ['even' if i % 2 == 0 else i for i in range(10)]
print(l)

get_odd_even = lambda x: 'even' if x % 2 == 0 else 'odd'
print(get_odd_even(1))
print(get_odd_even(2))


