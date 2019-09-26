# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-map-list-iterator/
Created on Wed Oct 24 12:34:28 2018

@author: Akitaka
"""

i = '1 2 3'
print(map(int, i.split(' ')))
print(list(map(int, i.split(' '))))
print([int(x) for x in i.split(' ')])

i2 = ['1 2 3', '4 5 6', '7 8 9']
input_array = []
for data in i2:
    #input_array.append(list(map(int, data.split(' '))))
    input_array.append([int(x) for x in data.split(' ')])

print(input_array)
