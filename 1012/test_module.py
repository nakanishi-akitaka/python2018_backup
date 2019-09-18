# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-if-name-main/
Created on Fri Oct 12 12:56:42 2018

@author: Akitaka
"""

def func():
    print('    This is func() in test_module.py')
    print('    __name__ is', __name__)

if __name__ == '__main__':
    print("Start if __name__ == '__main__'")
    print('call func()')
    func()