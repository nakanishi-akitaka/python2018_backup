# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-pprint-pretty-print/
Created on Fri Oct 12 14:05:41 2018

@author: Akitaka
"""

import pprint

l = [{'Name': 'Alice XXX', 'Age': 40, 'Points': [80, 20]}, 
     {'Name': 'Bob YYY', 'Age': 20, 'Points': [90, 10]},
     {'Name': 'Charlie ZZZ', 'Age': 30, 'Points': [70, 30]}]

print(l)
print()
pprint.pprint(l)
print()
pprint.pprint(l, width=40)
print()
pprint.pprint(l, width=200)
print()
pprint.pprint(l, depth=1)
print()
pprint.pprint(l, depth=2)
print()
pprint.pprint(l, indent=4, width=4)

l_long = [list(range(10)), list(range(100, 110))]

print()
print(l_long)
print()
pprint.pprint(l_long, width=40)
print()
pprint.pprint(l_long, width=40, compact=True)

print()
s_normal = str(l)
print(s_normal)
print(type(s_normal))

print()
s_pp = pprint.pformat(l)
print(s_pp)
print(type(s_pp))

l_2d = [list(range(10)), list(range(10)), list(range(10))]

print()
print(l_2d)
print()
pprint.pprint(l_2d)




