# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-str-replace-translate-re-sub/
Created on Fri Oct 26 11:11:30 2018

@author: Akitaka
"""

s = 'one two one two one'
print(s.replace(' ', '-'))
print(s.replace(' ', ''))
print(s.replace('one', 'XXX'))
print(s.replace('one', 'XXX', 2))
print(s.replace('one', 'XXX').replace('two', 'YYY'))
print(s.replace('one', 'XtwoX').replace('two', 'YYY'))
print(s.replace('two', 'YYY').replace('one', 'XtwoX'))

#%%
s_lines = 'one\ntwo\nthree'
print(s_lines)
print(s_lines.replace('\n', '-'))

s_lines_multi = 'one\ntwo\r\nthree'
print(s_lines_multi)
print(s_lines_multi.replace('\r\n', '-').replace('\n', '-'))
print(s_lines_multi.replace('\n', '-').replace('\r\n', '-'))

print(s_lines_multi.splitlines())
print('-'.join(s_lines_multi.splitlines()))

#%%
s = 'one two one two one'
print(s.translate(str.maketrans({'o': 'O', 't': 'T'})))
print(s.translate(str.maketrans({'o': 'XXX', 't': None})))
print(s.translate(str.maketrans('ow', 'XY', 'n')))

#%%

import re 
s = 'aaa@xxx.com bbb@yyy.com ccc@zzz.com'
print(re.sub('[a-z]*@', 'ABC@', s))
print(re.sub('[a-z]*@', 'ABC@', s, 2))

print(re.sub('[xyz]', '1', s))
print(re.sub('aaa|bbb|ccc', 'ABC', s))

print(re.sub('([a-z]*)@', '\\1-123@', s))
print(re.sub('([a-z]*)@', r'\1-123@', s))

t = re.subn('[a-z]*@', 'ABC@', s)
print(t)
print(type(t))
print(t[0])
print(t[1])

s = 'abcdefghij'
print(s[:4] + 'XXX' + s[7:])
print(s[:4] + '-' + s[7:])
print(s[:4] + '+++++' + s[4:])


