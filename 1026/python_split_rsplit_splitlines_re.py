# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-split-rsplit-splitlines-re/
Created on Fri Oct 26 11:21:33 2018

@author: Akitaka
"""
s_blank = 'one two     three\nfour\tfive'
print(s_blank)
print(s_blank.split())
print(type(s_blank.split()))

s_comma = 'one,two,three,four,five'
print(s_comma.split(','))
print(s_comma.split('three'))
print(s_comma.split(',', 2))

s_lines = 'one\ntwo\nthree\nfour'
print(s_lines)
print(s_lines.split('\n', 1))
print(s_lines.split('\n', 1)[0])
print(s_lines.split('\n', 1)[1])
print(s_lines.split('\n', 1)[-1])
print(s_lines.split('\n', 2)[-1])

#%%
print(s_lines.rsplit('\n', 1))
print(s_lines.rsplit('\n', 1)[0])
print(s_lines.rsplit('\n', 1)[1 ])
print(s_lines.rsplit('\n', 2)[0])

#%%
s_lines_multi = '1 one\n2 two\r\n3 three\n'
print(s_lines_multi)
print(s_lines_multi.split())
print(s_lines_multi.split('\n'))
print(s_lines_multi.splitlines())
print(s_lines_multi.splitlines(True))

#%%
import re
s_nums = 'one1two22three333four'
print(re.split('\d+', s_nums))
print(re.split('\d+', s_nums, 2))

#%%
s_marks = 'one-two+three#four'
print(re.split('[-+#]', s_marks))
s_strs = 'oneXXXtwoYYYthreeZZZfour'
print(re.split('XXX|YYY|ZZZ', s_strs))

#%%
l = ['one', 'two', 'three']
print(','.join(l))
print('\n'.join(l))
print(''.join(l))

#%%
s = 'abcdefghij'
print(s[:5])
print(s[5:])

#%%
s_tuple = s[:5], s[5:]
print(s_tuple)
print(type(s_tuple))

s_first, s_last = s[:5], s[5:]
print(s_first)
print(s_last)

s_first, s_second, s_last = s[:3], s[3:6], s[6:]
print(s_first)
print(s_second)
print(s_last)

half = len(s) // 2
print(half)

s_first, s_last = s[:half], s[half:]
print(s_first)
print(s_last)

print(s_first + s_last)

