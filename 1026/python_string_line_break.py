# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-string-line-break/
Created on Fri Oct 26 11:42:14 2018

@author: Akitaka
"""

s = "Line1\nLine2\nLine3"
print(s)

s = "Line1\r\nLine2\r\nLine3"
print(s)

s = '''Line1
Line2
Line3'''
print(s)

s = '''
    Line1
    Line2
    Line3
    '''
print(s)

s = 'Line1\n'\
    'Line2\n'\
    'Line3'
print(s)

s = 'Line1\n'\
    '    Line2\n'\
    '        Line3'
print(s)

s = ('Line1\n'
     'Line2\n'
     'Line3')
print(s)
s = ('Line1\n'
     '    Line2\n'
     '        Line3')
print(s)

s = '''\
Line1
Line2
Line3'''
print(s)

s = '''\
Line1
    Line2
        Line3'''
print(s)

#%%
l = ['Line1', 'Line2', 'Line3']
s = '\n'.join(l)
print(s)

s = 'Line1\nLine2\r\nLine3'
l = s.splitlines()
print(l)

#%%
s = 'Line1\nLine2\r\nLine3'
s_new = ''.join(s.splitlines())
print(s_new)
s_new = ' '.join(s.splitlines())
print(s_new)
s_new = ','.join(s.splitlines())
print(s_new)
s_new = '\r\n'.join(s.splitlines())
print(s_new)

s = 'Line1\nLine2\nLine3'
s_new = s.replace('\n', '')
print(s_new)
s_new = s.replace('\n', ',')
print(s_new)

s = 'Line1\nLine2\r\nLine3'
s_new = s.replace('\n', ',')
print(s_new)
s_new = s.replace('\r\n', ',')
print(s_new)

s = 'Line1\nLine2\r\nLine3'
s_new = s.replace('\r\n', ',').replace('\n', ',')
print(s_new)
s_new = s.replace('\n', ',').replace('\r\n', ',')
print(s_new)
s_new = ','.join(s.splitlines())
print(s_new)

s = 'aaa\n'
print(s + 'bbb')
s_new = s.rstrip()
print(s_new + 'bbb')

#%%
print('a')
print('b')
print('c')

print('a', end='')
print('b', end='')
print('c', end='')

print('a', end='-')
print('b', end='-')
print('c')