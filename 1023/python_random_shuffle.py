# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-random-shuffle/
Created on Tue Oct 23 12:52:55 2018

@author: Akitaka
"""

import random
l = list(range(5))
print(l)

random.shuffle(l)
print(l)

l = list(range(5))
print(l)

lr = random.sample(l, len(l))
print(lr)
print(l)

#%%
s = 'abcde'
print(s)

t = tuple(range(5))
print(t)

sr = ''.join(random.sample(s, len(s)))
print(sr)

tr = tuple(random.sample(t, len(t)))
print(tr)

#%%
random.seed(0)
l = list(range(5))
random.shuffle(l)
print(l)

random.seed(0)
random.shuffle(l)
print(l)

random.seed(0)
random.shuffle(l)
print(l)

random.seed(0)
random.shuffle(l)
print(l)
