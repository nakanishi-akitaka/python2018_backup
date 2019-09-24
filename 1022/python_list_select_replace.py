# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-list-select-replace/
Created on Mon Oct 22 16:13:21 2018

@author: Akitaka
"""
l = list(range(-5, 6))

print(l)

l_square = [i**2 for i in l]
print(l_square)

l_str = [str(i) for i in l]
print(l_str)

l_even = [i for i in l if i % 2 == 0]
print(l_even)

l_minus = [i for i in l if i < 0]
print(l_minus)

l_odd = [i for i in l if not i % 2 == 0]
print(l_odd)

l_minus_or_even = [i for i in l if (i < 0) or (i % 2 == 0)]
print(l_minus_or_even)

l_minus_and_odd = [i for i in l if (i < 0) and not (i % 2 == 0)]
print(l_minus_and_odd)

#%%

a = 80
x = 100 if a > 50 else 0
print(x)

b = 30
y = 100 if b > 50 else 0
print(y)

#%%

l_replace = [100 if i > 0 else i for i in l]
print(l_replace)

l_replace2 = [100 if i > 0 else 0 for i in l]
print(l_replace2)

l_convert = [i * 10 if i % 2 == 0 else i for i in l]
print(l_convert)

l_convert2 = [i * 10 if i % 2 == 0 else i / 10 for i in l]
print(l_convert2)







