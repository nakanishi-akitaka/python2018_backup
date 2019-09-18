# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-if-name-main/
Created on Fri Oct 12 12:57:06 2018

@author: Akitaka
"""

import test_module

print('This is test_main.py')
print('test_module.__name__ is', test_module.__name__)

print('---')
print('call test_module.func()')

test_module.func()  