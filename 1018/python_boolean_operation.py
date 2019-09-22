# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-boolean-operation/
Created on Thu Oct 18 11:28:10 2018

@author: Akitaka
"""

x = True
y = False
print(x and y)
print(x or y)
print(not x)

print(bool(10))
print(bool(0))
print(bool(''))
print(bool('0'))
print(bool('False'))
print(bool([]))
print(bool([False]))

x = 10 # True
y = 0 # False
print(x and y)
print(x or y)
print(not x)

x = 10 # True
y = 100 # True
print(x and y)
print(y and x)
print(x or y)
print(y or x)

x = 0 # False
y = 0.0 # False
print(x and y)
print(y and x)
print(x or y)
print(y or x)

#%%
def test():
    print('function is called')
    return True

print(True and test())
print(False and test())
print(True or test())
print(False or test())
