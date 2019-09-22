# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-bit-operation/
Created on Thu Oct 18 12:13:06 2018

@author: Akitaka
"""
x = 9  # 0b1001
y = 10 # 0b1010
print(x & y)
print(bin(x & y))

print(x | y)
print(bin(x | y))

print(x ^ y)
print(bin(x ^ y))

x = -9
print(x)
print(bin(x))

print(bin(x & 0xff))
print(format(x & 0xffff, 'x'))

x = 9
print(~x)
print(bin(~x))

print(bin(~x & 0xff))
print(format(~x & 0b1111, '04b'))

x = 9
print(x << 1)
print(bin(x << 1))
print(x >> 1)
print(bin(x >> 1))

x = -9
print(bin(x))
print(bin(x & 0xff))

print(x << 1)
print(bin(x << 1))
print(bin((x << 1) & 0xff))

print(x >> 1)
print(bin(x >> 1))
print(bin((x >> 1) & 0xff))
