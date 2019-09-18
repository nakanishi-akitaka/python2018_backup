# -*- coding: utf-8 -*-
"""
Pythonのprint関数で文字列、数値および変数の値を出力
https://note.nkmk.me/python-print-basic/
Created on Fri Oct 12 14:01:01 2018

@author: Akitaka
"""

i = 255

print('left   : {:<8}'.format(i))
print('center : {:^8}'.format(i))
print('right  : {:>8}'.format(i))
print('zero   : {:08}'.format(i))
print('bin    : {:b}'.format(i))
print('oct    : {:o}'.format(i))
print('hex    : {:x}'.format(i))
# left   : 255     
# center :   255   
# right  :      255
# zero   : 00000255
# bin    : 11111111
# oct    : 377
# hex    : ff

f = 0.1234

print('digit   : {:.2}'.format(f))
print('digit   : {:.6f}'.format(f))
print('exp     : {:.4e}'.format(f))
print('percent : {:.0%}'.format(f))
# digit   : 0.12
# digit   : 0.123400
# exp     : 1.2340e-01
# percent : 12%