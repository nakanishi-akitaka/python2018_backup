# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-arithmetic-operator/
Created on Thu Oct 18 11:16:16 2018

@author: Akitaka
"""

print(10 + 3)

print(10 - 3)

print(10 * 3)

print(10 / 3)

print(10 // 3)

print(10 % 3)

print(10 ** 3)

print (2 ** 0.5)

print(10 ** -2)

print(0 ** 0)

#print(10 / 0)
#print(10 // 0)
#print(10 % 0)
#print(0 ** -1)

a = 10
b = 3
c = a + b

print('a:', a)
print('b:', b)
print('c:', c)

a = 10
b = 3
a += b
print('a:', a)
print('b:', b)

a = 10
b = 3
a %= b
print('a:', a)
print('b:', b)


a = 10
b = 3
a **= b
print('a:', a)
print('b:', b)

print( 2 + 3.0)
print(type(2 + 3.0))

print(10 / 2)
print(type(10 / 2))

print(2 ** 3)
print(type(2 ** 3))

print(2.0 ** 3)
print(type(2.0 ** 3))

print(25 ** 0.5)
print(type(25 ** 0.5))

print(100 / 10 ** 2 + 2 * 3 - 5)
print(100 / (10 ** 2) + (2 * 3) - 5)
print((100 / 10) ** 2 + 2 * (3 - 5))

a_l = [0, 1, 2]
b_l = [10, 20, 30]
a_t = (0, 1, 2)
b_t = (10, 20, 30)
a_s = 'abc'
b_s = 'xyz'

print(a_l + b_l)
print(a_t + b_t)
print(a_s + b_s)

#print(a_l + 3)
print(a_l + [3])
#print(a_t + 3)
print(a_t + (3, ))

a_l += b_l
print(a_l)

a_t += b_t
print(a_t)

a_s += b_s
print(a_s)

print(b_l * 3)
print(3 * b_l)
print(b_t * 3)
print(3 * b_t)
print(b_s * 3)
print(3 * b_s)
#print(b_l * 0.5)
print(b_l * -1)

b_l *= 3
print(b_l)
b_t *= 3
print(b_t)
b_s *= 3
print(b_s)

a_l = [0, 1, 2]
b_l = [10, 20, 30]
c_l = a_l + b_l * 3
print(c_l)