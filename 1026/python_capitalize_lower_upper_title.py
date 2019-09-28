# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-capitalize-lower-upper-title/
Created on Fri Oct 26 12:23:18 2018

@author: Akitaka
"""

str_org = "pYthon iS a gooD proGramming laNguage"
print(str_org.capitalize())
print(str_org)

str_new = str_org.capitalize()
str_org = "Python is a good programming language"
print(str_org.lower())

str_org = "Python is a good programming language"
print(str_org.upper())

str_org = "Python is a good programming language"
print(str_org.title())

str_org = "pYthon iS a gooD proGramming laNguage"
print(str_org.swapcase())

#%%
print('Python'.islower())
print('python'.islower())
print('ｐｙｔｈｏｎ'.islower())
print('python パイソン 123'.islower())
print('パイソン 123'.islower())
print(''.islower())


print('PYTHON'.isupper())
print('Python'.isupper())
print('ＰＹＴＨＯＮ'.islower())
print('PYTHON パイソン 123'.isupper())
print('パイソン 123'.isupper())


