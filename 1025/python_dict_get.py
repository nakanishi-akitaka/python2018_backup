# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-dict-get/
Created on Thu Oct 25 11:08:22 2018

@author: Akitaka
"""
d = {'key1': 'val1', 'key2': 'val2', 'key3': 'val3'}
print(d['key1'])

d['key4'] = 'val4'
print(d)

print(d.get('key1'))
print(d.get('key5'))
print(d.get('key5', 'NO KEY'))
print(d.get('key5', 100))
print(d)
