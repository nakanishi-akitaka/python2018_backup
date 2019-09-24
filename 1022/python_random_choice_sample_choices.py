# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-random-choice-sample-choices/
Created on Mon Oct 22 16:24:02 2018

@author: Akitaka
"""

import random

l = list(range(5))
print(l)
print(random.choice(l))
print(random.choice(['G', 'C', 'P']))
print(random.choice(['Daikichi', 'Kichi', 'Kyou']))

print(random.sample(l, 3))
print(random.choices(l, k=3))
print(random.choices(l, k=10))

print(random.choices(l, k=3, weights=[1, 1, 1, 10, 1]))
print(random.choices(l, k=3, weights=[1, 1, 0, 0, 0]))
print(random.choices(l, k=3, cum_weights=[1, 2, 3, 13, 14]))

random.seed(0)
print(random.choice(l))