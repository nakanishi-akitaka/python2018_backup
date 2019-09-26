# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-itertools-product/
Created on Wed Oct 24 12:00:56 2018

@author: Akitaka
"""

import itertools
import pprint


l1 = ['a', 'b', 'c']
l2 = ['X', 'Y', 'Z']

p = itertools.product(l1, l2)
print(p)
print(type(p))

for v in p:
    print(v)

for v1, v2 in itertools.product(l1, l2):
    print(v1, v2)

for v1 in l1:
    for v2 in l2:
        print(v1, v2)


l_p = list(itertools.product(l1, l2))
pprint.pprint(l_p)

print(type(l_p))
print(type(l_p[0]))


t = ('one', 'two')
d = {'key1': 'value1', 'key2': 'value2'}
r = range(2)

l_p = list(itertools.product(t, d, r))
pprint.pprint(l_p)

#%%
l1 = ['a', 'b']
pprint.pprint(list(itertools.product(l1, repeat=3)))
pprint.pprint(list(itertools.product(l1, l1, l1)))

l1 = ['a', 'b']
l2 = ['X', 'Y']
pprint.pprint(list(itertools.product(l1, l2, repeat=2)))
pprint.pprint(list(itertools.product(l1, l2, l1, l2)))

#%%
for v1, v2 in itertools.product(l1, l2):
    print(v1, v2)

for v1 in l1:
    for v2 in l2:
        print(v1, v2)



