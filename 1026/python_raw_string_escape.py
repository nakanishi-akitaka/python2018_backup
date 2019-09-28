# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-raw-string-escape/
Created on Fri Oct 26 10:47:11 2018

@author: Akitaka
"""

s = 'a\tb\nA\tB'
print(s)

rs = r'a\tb\nA\tB'
print(rs)
print(type(rs))
print(rs == 'a\\tb\\nA\\tB')

print(len(s))
print(list(s))
print(len(rs))
print(list(rs))

#%%
path = 'C:\\Windows\\system32\\cmd.exe'
rpath = r'C:\Windows\system32\cmd.exe'
print(path == rpath)

path2 = 'C:\\Windows\\system32\\'
rpath2 = r'C:\Windows\system32' + '\\'
print(path2 == rpath2)

#%%
s_r = repr(s)
print(s_r)
print(list(s_r))

s_r2 = repr(s)[1:-1]
print(s_r2)
print(s_r2 == rs)
print(r'\t' == repr('\t')[1:-1])
print(r'\\')
