# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-string-concat/
Created on Fri Oct 26 11:32:18 2018

@author: Akitaka
"""
s = 'aaa' + 'bbb' + 'ccc'
print(s)

s1 = 'aaa'
s2 = 'bbb'
s3 = 'ccc'
s = s1 + s2 + s3
print(s)

s = s1 + s2 + s3 + 'ddd'
print(s)

s1 += s2
print(s1)

s = 'aaa'
s += 'xxx'
print(s)

#%%
s = 'aaa''bbb''ccc'
print(s)

s = 'aaa'  'bbb'    'ccc'
print(s)

s = 'aaa'\
    'bbb'\
    'ccc'
print(s)

s1 = 'aaa'
s2 = 'bbb'

i = 100
f = 0.25

s = s1 + '_' + str(i) + '_' + s2 + '_' + str(f)
print(s)

s = s1 + '_' + format(i, '05') + '_' + s2 + '_' + format(f, '.5f')
print(s)
s = '{}_{:05}_{}_{:.5f}'.format(s1, i, s2, f)
print(s)

#%%
l = ['aaa', 'bbb', 'ccc']

s = ''.join(l)
print(s)

s = ','.join(l)
print(s)

s = '-'.join(l)
print(s)

s = '\n'.join(l)
print(s)

l = [2017, 12, 31]
s = '-'.join([str(n) for n in l])
print(s)

