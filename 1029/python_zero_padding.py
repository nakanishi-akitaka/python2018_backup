# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-zero-padding/
Created on Mon Oct 29 12:28:06 2018

@author: Akitaka
"""

s = '1234'
s_zero = s.zfill(8)
print(s_zero)
print(type(s_zero))

print(s.zfill(3))

#%%
s = '-1234'
print(s.zfill(8))
s = '+1234'
print(s.zfill(8))
s = 'abcd'
print(s.zfill(8))

#%%
n = 1234
print(type(n))
print(str(n).zfill(8))

#%%
s = '1234'
print(s.rjust(8, '0'))
print(s.ljust(8, '0'))
print(s.center(8, '0'))

s = '-1234'
print(s.rjust(8, '0'))
print(s.ljust(8, '0'))
print(s.center(8, '0'))

#%%
s = '1234'
print(format(s, '0>8'))
print('Zero Padding: {:0>8}'.format(s))


s = '-1234'
print(format(s, '0>8'))
print('Zero Padding: {:0>8}'.format(s))

print(format(int(s), '08'))
print('Zero Padding: {:08}'.format(int(s)))

n = 1234

print(format(n, '08'))
# 00001234

print('Zero Padding: {:08}'.format(n))
# Zero Padding: 00001234

print(format(n, '08x'))
# 000004d2

print('Zero Padding: {:08x}'.format(n))
# Zero Padding: 000004d2

n = -1234

print(format(n, '08'))
# -0001234

print('Zero Padding: {:08}'.format(n))
# Zero Padding: -0001234

print(f'Zero Padding: {n:08}')

#%%
n = 1234
print('Zero Padding: %08d' % n)
# Zero Padding: 00001234

n = -1234
print('Zero Padding: %08d' % n)
# Zero Padding: -0001234

s = '1234'
print('Zero Padding: %08s' % s)
# Zero Padding:     1234
print('Zero Padding: %08d' % int(s))
# Zero Padding: 00001234








