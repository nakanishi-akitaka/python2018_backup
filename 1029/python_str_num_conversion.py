# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-str-num-conversion/
Created on Mon Oct 29 16:03:01 2018

@author: Akitaka
"""
print(int('100'))
print(type(int('100')))
# 100
# <class 'int'>

# print(int('1.23'))
# ValueError: invalid literal for int() with base 10: '1.23'

# print(int('10,000'))
# ValueError: invalid literal for int() with base 10: '10,000'


print(int('10,000'.replace(',', '')))
# 10000

#%%
print(float('1.23'))
print(type(float('1.23')))
# 1.23
# <class 'float'>

print(float('.23'))
# 0.23

print(float('100'))
print(type(int('100')))
# 100.0
# <class 'int'>

#%%
print(int('101', 2))
print(int('70', 8))
print(int('FF', 16))
# 5
# 56
# 255

print(int('0b101', 0))
print(int('0o70', 0))
print(int('0xFF', 0))
# 5
# 56
# 255

#%%
print(float('1.23e-4'))
print(type(float('1.23e-4')))
# 0.000123
# <class 'float'>

print(float('1.23e4'))
print(type(float('1.23e4')))
# 12300.0
# <class 'float'>

#%%
print(int('１００'))
print(type(int('１００')))
# 100
# <class 'int'>

print(float('１００'))
print(type(float('１００')))
# 100.0
# <class 'float'>

# print(float('ー１．２３'))
# ValueError: could not convert string to float: '１．２３'

print(float('-１.２３'))
# -1.23

print(float('ー１．２３'.replace('ー', '-').replace('．', '.')))
# -1.23

#%%
import unicodedata

print(unicodedata.numeric('五'))
print(type(unicodedata.numeric('五')))
# 5.0
# <class 'float'>

print(unicodedata.numeric('十'))
# 10.0

print(unicodedata.numeric('参'))
# 3.0

print(unicodedata.numeric('億'))
# 100000000.0

# print(unicodedata.numeric('五十'))
# TypeError: numeric() argument 1 must be a unicode character, not str

# print(unicodedata.numeric('漢'))
# ValueError: not a numeric character


