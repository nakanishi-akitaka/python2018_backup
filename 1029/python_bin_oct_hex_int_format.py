# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-bin-oct-hex-int-format/
Created on Mon Oct 29 15:59:17 2018

@author: Akitaka
"""

bin_num = 0b10
oct_num = 0o10
hex_num = 0x10

print(bin_num)
print(oct_num)
print(hex_num)
# 2
# 8
# 16


Bin_num = 0B10
Oct_num = 0O10
Hex_num = 0X10

print(Bin_num)
print(Oct_num)
print(Hex_num)
# 2
# 8
# 16

#%%
print(type(bin_num))
print(type(oct_num))
print(type(hex_num))
# <class 'int'>
# <class 'int'>
# <class 'int'>

print(type(Bin_num))
print(type(Oct_num))
print(type(Hex_num))
# <class 'int'>
# <class 'int'>
# <class 'int'>

#%%
result = 0b10 * 0o10 + 0x10
print(result)
# 32

print(0b111111111111 is 0b1_1_1_1_1_1_1_1_1_1_1_1)
# True

bin_num = 0b1111_1111_1111
print(bin_num)
# 4095

i = 255

bin_str = bin(i)
oct_str = oct(i)
hex_str = hex(i)

print(bin_str)
print(oct_str)
print(hex_str)
# 0b11111111
# 0o377
# 0xff

print(type(bin_str))
print(type(oct_str))
print(type(hex_str))
# <class 'str'>
# <class 'str'>
# <class 'str'>

#%%
bin_str = format(i, 'b')
oct_str = format(i, 'o')
hex_str = format(i, 'x')

print(bin_str)
print(oct_str)
print(hex_str)
# 11111111
# 377
# ff

print(type(bin_str))
print(type(oct_str))
print(type(hex_str))
# <class 'str'>
# <class 'str'>
# <class 'str'>
#%%
bin_str = format(i, '#b')
oct_str = format(i, '#o')
hex_str = format(i, '#x')

print(bin_str)
print(oct_str)
print(hex_str)
# 0b11111111
# 0o377
# 0xff

#%%
bin_str = format(i, '08b')
oct_str = format(i, '08o')
hex_str = format(i, '08x')

print(bin_str)
print(oct_str)
print(hex_str)
# 11111111
# 00000377
# 000000ff

bin_str = format(i, '#010b')
oct_str = format(i, '#010o')
hex_str = format(i, '#010x')

print(bin_str)
print(oct_str)
print(hex_str)
# 0b11111111
# 0o00000377
# 0x000000ff


#%%
print('bin is {:08b}'.format(i))
print('oct is {:08o}'.format(i))
print('hex is {:08x}'.format(i))
# bin is 11111111
# oct is 00000377
# hex is 000000ff

#%%
x = -9

print(x)
print(bin(x))
# -9
# -0b1001

print(bin(x & 0xff))
print(format(x & 0xffff, 'x'))
# 0b11110111
# fff7

#%%
dec_num = int('10')
bin_num = int('10', 2)
oct_num = int('10', 8)
hex_num = int('10', 16)

print(dec_num)
print(bin_num)
print(oct_num)
print(hex_num)
# 10
# 2
# 8
# 16

print(type(dec_num))
print(type(bin_num))
print(type(oct_num))
print(type(hex_num))
# <class 'int'>
# <class 'int'>
# <class 'int'>
# <class 'int'>

#%%
dec_num = int('10', 0)
bin_num = int('0b10', 0)
oct_num = int('0o10', 0)
hex_num = int('0x10', 0)

print(dec_num)
print(bin_num)
print(oct_num)
print(hex_num)
# 10
# 2
# 8
# 16

Bin_num = int('0B10', 0)
Oct_num = int('0O10', 0)
Hex_num = int('0X10', 0)

print(Bin_num)
print(Oct_num)
print(Hex_num)
# 2
# 8
# 16

#%%
a = '0b1001'
b = '0b0011'

c = int(a, 0) + int(b, 0)

print(c)
print(bin(c))
# 12
# 0b1100

#%%
a_0b = '0b1110001010011'

a_0x = format(int(a, 0), '#010x')
a_0o = format(int(a, 0), '#010o')
print(a_0x)
print(a_0o)
# 0x00000009
# 0o00000011


