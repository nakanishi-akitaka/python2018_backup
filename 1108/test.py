# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 15:18:03 2018

@author: Akitaka
"""
# https://qiita.com/croissant1028/items/94c4b7fd360cfcdef0e4
import functools

print (functools.reduce(lambda x, y: x + y, range(1, 5)))

# http://python-remrin.hatenadiary.jp/entry/2017/05/13/221530
import numpy as np

a1 = np.arange(4)
print(a1.repeat(2))
#  [0 0 1 1 2 2 3 3]

print(a1.repeat([3, 2, 1, 0]))
# [0 0 0 1 1 2]