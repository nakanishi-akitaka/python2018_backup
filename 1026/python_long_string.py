# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-long-string/
Created on Fri Oct 26 12:17:20 2018

@author: Akitaka
"""

n = 1 + 2 \
   + 3
print(n)

s = 'aaa' 'bbb'
print(s)

s = 'https://ja.wikipedia.org/wiki/'\
    '%E3%83%97%E3%83%AD%E3%82%B0%E3%83'\
    '%A9%E3%83%9F%E3%83%B3%E3%82%B0%E8%A8%80%E8%AA%9E'
print(s)

s_var = 'xxx'

# s = 'aaa' s_var 'bbb'
# SyntaxError: invalid syntax

s = 'aaa' + s_var + 'bbb'

print(s)

s = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'\
    + s_var\
    + 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'

print(s)

#%%
s = ('https://ja.wikipedia.org/wiki/'
     '%E3%83%97%E3%83%AD%E3%82%B0%E3%83'
     '%A9%E3%83%9F%E3%83%B3%E3%82%B0%E8%A8%80%E8%AA%9E')

print(s)


s = ('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
     + s_var
     + 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')

print(s)

