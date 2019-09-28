# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-str-literal-constructor/
Created on Fri Oct 26 10:37:51 2018

@author: Akitaka
"""
s = 'abc'
print(s)
print(type(s))

s = "abc"
print(s)
print(type(s))

s_sq = 'abc'
s_dq = "abc"
print(s_sq == s_dq)

s_sq = 'a\'b"c'
print(s_sq)

s_sq = 'a\'b\"c'
print(s_sq)

s_dq = "a'b\"c"
print(s_dq)

s_dq = "a\'b\"c"
print(s_dq)

s_sq = 'a\'b"c'
s_dq = "a'b\"c"
print(s_sq == s_dq)

#%%
s = 'abc\nxyz'
print(s)

s_tq = '''abc
xyz'''

print(s_tq)
print(type(s_tq))

s_tq = '''abc'''
print(s_tq)

s_tq_sq = '''\'abc\'
"xyz"'''
print(s_tq_sq)

s_tq_dq = """'abc'
\"xyz\""""

print(s_tq_dq)
print(s_tq_sq == s_tq_dq)

#%%
s_tq = '''abc
          xyz'''
print(s_tq)

s_multi = ('abc\n'
           'xyz')
print(s_multi)

#%%
i = 100

s_i = str(i)
print(s_i)
print(type(s_i))

f = 0.123
s_f = str(f)
print(s_f)
print(type(s_f))

i = 0xFF
print(i)
s_i = str(i)
print(s_i)

f = 1.23e+10
print(f)
s_f = str(f)
print(s_f)

s_i_format = format(i, '#X')
print(s_i_format)

s_f_format = format(f, '.2e')
print(s_f_format)

#%%
l = [0, 1, 2]
s_l = str(l)
print(s_l)
print(type(s_l))

d = {'a': 1,
     'b': 2,
     'c': 3}
s_d = str(d)
print(s_d)
print(type(s_d))


#%%
import pprint

dl = {'a': 1, 'b': 2, 'c': [100, 200, 300]}
s_dl = str(dl)
print(s_dl)

p_dl = pprint.pformat(dl, width=10)
print(p_dl)
print(type(p_dl))
