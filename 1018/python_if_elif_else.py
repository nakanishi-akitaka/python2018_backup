# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-if-elif-else/
Created on Thu Oct 18 13:20:31 2018

@author: Akitaka
"""

def if_test(num):
    if num > 100:
        print('100 < num')
    elif num > 50:
        print('50 < num <= 100')
    elif num > 0:
        print('0 < num <= 50')
    elif num == 0:
        print('num == 0')
    else:
        print('num < 0')

if_test(1000)
if_test(70)
if_test(0)
if_test(-100)

def if_test2(num):
    if 50 < num < 100:
        print('50 < num < 100')
    else:
        print('num <=  50 or num >= 100')

if_test2(70)
if_test2(0)

i = 10
print(type(i))
f = 10.0
print(type(f))
print(i == f)
print(i is f)

def if_test_in(s):
    if 'a' in s:
        print('a is in string')
    else:
        print('a in NOT in string')

if_test_in('apple')
if_test_in('melon')

if 10:
    print('True')

if [0, 1, 2]:
    print('True')

print(bool(10))
print(bool(0.0))
print(bool([]))
print(bool('False'))

def if_test_list(l):
    if l:
        print('list is NOT empty')
    else:
        print('list is empty')

if_test_list([0, 1, 2])
if_test_list([])

def if_test_and_not(num):
    if num >= 0 and not num % 2 == 0:
        print('num is positive odd')
    else:
        print('num is NOT positive odd')

if_test_and_not(5)
if_test_and_not(10)
if_test_and_not(-10)

def if_test_and_not_or(num):
    if num >= 0 and not num % 2 == 0 or num == -10:
        print('num is positive odd or -10')
    else:
        print('num is NOT positive odd or -10')

if_test_and_not_or(5)
if_test_and_not_or(10)
if_test_and_not_or(-10)

def if_test_and_backslash(num):
    if num >= 0 \
        and not num % 2 == 0:
        print('num is positive odd')
    else:
        print('num is NOT positive odd')

if_test_and_backslash(5)


def if_test_and_brackets(num):
    if (num >= 0 
        and not num % 2 == 0):
        print('num is positive odd')
    else:
        print('num is NOT positive odd')

if_test_and_brackets(5)

