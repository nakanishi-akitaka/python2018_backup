# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-random-randrange-randint/
Created on Tue Oct 30 11:12:30 2018

@author: Akitaka
"""
import random

print(random.random())
# 0.447948002492365

import random

print(random.uniform(100, 200))
# 175.26585048238275

print(random.uniform(100, -100))
# -27.82338731501028

print(random.uniform(100, 100))
# 100.0

print(random.uniform(1.234, 5.637))
# 2.606743596829249


#%%
import random

print(list(range(10)))
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(random.randrange(10))
# 5

print(list(range(10, 20, 2)))
# [10, 12, 14, 16, 18]

print(random.randrange(10, 20, 2))
# 18

#%%
print(random.randint(50, 100))
# print(random.randrange(50, 101))
# 74

#%%
import random

print([random.random() for i in range(5)])
# [0.5518201298350598, 0.3476911314933616, 0.8463426180468342, 0.8949046353303931, 0.40822657702632625]

#%%
print([random.randint(0, 10) for i in range(5)])
# [8, 5, 7, 10, 7]

print(random.sample(range(10), k=5))
# [6, 4, 3, 7, 5]

print(random.sample(range(100, 200, 10), k=5))
# [130, 190, 140, 150, 170]

#%%
random.seed(0)
print(random.random())
# 0.8444218515250481