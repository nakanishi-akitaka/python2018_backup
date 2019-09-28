# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-str-len/
Created on Fri Oct 26 10:58:44 2018

@author: Akitaka
"""

s = 'abcde'
print(len(s))

s_length = len(s)
print(s_length)
print(type(s_length))

s = 'あいうえお'
print(len(s))

s = 'abcdeあいうえお'
print(len(s))

s = 'a\tb\\c'
print(s)
print(len(s))

s = r'a\tb\\c'
print(s)
print(len(s))

s = r'\u3042\u3044\u3046'
print(s)
print(len(s))

s = 'a\nb'
print(s)
print(len(s))

s = 'a\r\nb'
print(s)
print(len(s))

#%%
s = 'abc\nabcd\r\nab'
print(s)
print(len(s))
print(s.splitlines())
print(len(s.splitlines()))
print([len(line) for line in s.splitlines()])
print(sum(len(line) for line in s.splitlines()))


