# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-round-decimal-quantize/
Created on Tue Oct 30 10:52:42 2018

@author: Akitaka
"""

f = 123.456

print(round(f))
# 123

print(type(round(f)))
# <class 'int'>

print(round(f, 1))
# 123.5

print(round(f, 2))
# 123.46

print(round(f, -1))
# 120.0

print(round(f, -2))
# 100.0

print(round(f, 0))
# 123.0

print(type(round(f, 0)))
# <class 'float'>

#%%
i = 99518

print(round(i))
# 99518

print(round(i, 2))
# 99518

print(round(i, -1))
# 99520

print(round(i, -2))
# 99500

print(round(i, -3))
# 100000

#%%
print('0.4 =>', round(0.4))
print('0.5 =>', round(0.5))
print('0.6 =>', round(0.6))
# 0.4 => 0
# 0.5 => 0
# 0.6 => 1

print('4 =>', round(4, -1))
print('5 =>', round(5, -1))
print('6 =>', round(6, -1))
# 4 => 0
# 5 => 0
# 6 => 10

print('0.5 =>', round(0.5))
print('1.5 =>', round(1.5))
print('2.5 =>', round(2.5))
print('3.5 =>', round(3.5))
print('4.5 =>', round(4.5))
# 0.5 => 0
# 1.5 => 2
# 2.5 => 2
# 3.5 => 4
# 4.5 => 4

print('0.05 =>', round(0.05, 1))
print('0.15 =>', round(0.15, 1))
print('0.25 =>', round(0.25, 1))
print('0.35 =>', round(0.35, 1))
print('0.45 =>', round(0.45, 1))
# 0.05 => 0.1
# 0.15 => 0.1
# 0.25 => 0.2
# 0.35 => 0.3
# 0.45 => 0.5

#%%
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
print(Decimal(0.05))
# 0.05000000000000000277555756156289135105907917022705078125

print(type(Decimal(0.05)))
# <class 'decimal.Decimal'>

print(Decimal(0.5))
# 0.5

print(Decimal('0.05'))
# 0.05

#%%
f = 123.456

print(Decimal(str(f)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
# 123

print(Decimal(str(f)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))
# 123.5

print(Decimal(str(f)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
# 123.46

print('0.4 =>', Decimal(str(0.4)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
print('0.5 =>', Decimal(str(0.5)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
print('0.6 =>', Decimal(str(0.6)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
# 0.4 => 0
# 0.5 => 1
# 0.6 => 1

print('0.05 =>', round(0.05, 1))
print('0.15 =>', round(0.15, 1))
print('0.25 =>', round(0.25, 1))
print('0.35 =>', round(0.35, 1))
print('0.45 =>', round(0.45, 1))
# 0.05 => 0.1
# 0.15 => 0.1
# 0.25 => 0.2
# 0.35 => 0.3
# 0.45 => 0.5

print('0.05 =>', Decimal(0.05).quantize(Decimal('0.1'), rounding=ROUND_HALF_EVEN))
print('0.15 =>', Decimal(0.15).quantize(Decimal('0.1'), rounding=ROUND_HALF_EVEN))
print('0.25 =>', Decimal(0.25).quantize(Decimal('0.1'), rounding=ROUND_HALF_EVEN))
print('0.35 =>', Decimal(0.35).quantize(Decimal('0.1'), rounding=ROUND_HALF_EVEN))
print('0.45 =>', Decimal(0.45).quantize(Decimal('0.1'), rounding=ROUND_HALF_EVEN))
# 0.05 => 0.1
# 0.15 => 0.1
# 0.25 => 0.2
# 0.35 => 0.3
# 0.45 => 0.5

print('0.05 =>', Decimal(str(0.05)).quantize(Decimal('0.1'), rounding=ROUND_HALF_EVEN))
print('0.15 =>', Decimal(str(0.15)).quantize(Decimal('0.1'), rounding=ROUND_HALF_EVEN))
print('0.25 =>', Decimal(str(0.25)).quantize(Decimal('0.1'), rounding=ROUND_HALF_EVEN))
print('0.35 =>', Decimal(str(0.35)).quantize(Decimal('0.1'), rounding=ROUND_HALF_EVEN))
print('0.45 =>', Decimal(str(0.45)).quantize(Decimal('0.1'), rounding=ROUND_HALF_EVEN))
# 0.05 => 0.0
# 0.15 => 0.2
# 0.25 => 0.2
# 0.35 => 0.4
# 0.45 => 0.4

print(Decimal(2.675))
# 2.67499999999999982236431605997495353221893310546875

print(Decimal(2.675).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
# 2.67

print(Decimal(str(2.675)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
# 2.68

print(Decimal(2.675).quantize(Decimal('0.01'), rounding=ROUND_HALF_EVEN))
# 2.67

print(Decimal(str(2.675)).quantize(Decimal('0.01'), rounding=ROUND_HALF_EVEN))
# 2.68

d = Decimal('123.456').quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

print(d)
# 123.46

print(type(d))
# <class 'decimal.Decimal'>

# print(1.2 + d)
# TypeError: unsupported operand type(s) for +: 'float' and 'decimal.Decimal'

print(1.2 + float(d))
# 124.66

#%%
i = 99518

print(Decimal(i).quantize(Decimal('10'), rounding=ROUND_HALF_UP))
# 99518

print(Decimal('10').as_tuple())
# DecimalTuple(sign=0, digits=(1, 0), exponent=0)

print(Decimal('1E1').as_tuple())
# DecimalTuple(sign=0, digits=(1,), exponent=1)

print(Decimal(i).quantize(Decimal('1E1'), rounding=ROUND_HALF_UP))
# 9.952E+4

print(int(Decimal(i).quantize(Decimal('1E1'), rounding=ROUND_HALF_UP)))
# 99520

print(int(Decimal(i).quantize(Decimal('1E2'), rounding=ROUND_HALF_UP)))
# 99500

print(int(Decimal(i).quantize(Decimal('1E3'), rounding=ROUND_HALF_UP)))
# 100000

print('4 =>', int(Decimal(4).quantize(Decimal('1E1'), rounding=ROUND_HALF_UP)))
print('5 =>', int(Decimal(5).quantize(Decimal('1E1'), rounding=ROUND_HALF_UP)))
print('6 =>', int(Decimal(6).quantize(Decimal('1E1'), rounding=ROUND_HALF_UP)))
# 4 => 0
# 5 => 10
# 6 => 10

#%%
def my_round(val, digit=0):
    p = 10 ** digit
    return (val * p * 2 + 1) // 2 / p

my_round_int = lambda x: int((x * 2 + 1) // 2)

#%%
print(int(my_round(f)))
# 123

print(my_round_int(f))
# 123

print(my_round(f, 1))
# 123.5

print(my_round(f, 2))
# 123.46

print(int(my_round(0.4)))
print(int(my_round(0.5)))
print(int(my_round(0.6)))
# 0
# 1
# 1

#%%
i = 99518

print(int(my_round(i, -1)))
# 99520

print(int(my_round(i, -2)))
# 99500

print(int(my_round(i, -3)))
# 100000

print(int(my_round(4, -1)))
print(int(my_round(5, -1)))
print(int(my_round(6, -1)))
# 0
# 10
# 10

#%%
print(int(my_round(-0.4)))
print(int(my_round(-0.5)))
print(int(my_round(-0.6)))
# 0
# 0
# -1

import math

def my_round2(val, digit=0):
    p = 10 ** digit
    s = math.copysign(1, val)
    return (s * val * p * 2 + 1) // 2 / p * s

print(int(my_round2(-0.4)))
print(int(my_round2(-0.5)))
print(int(my_round2(-0.6)))
# 0
# -1
# -1




