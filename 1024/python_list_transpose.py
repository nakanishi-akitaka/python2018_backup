# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-list-transpose/
Created on Wed Oct 24 10:33:19 2018

@author: Akitaka
"""

import numpy as np
import pandas as pd
l_2d = [[00, 1, 2], [3, 4, 5]]
arr_t = np.array(l_2d).T
print(arr_t)
print(type(arr_t))

l_2d_t = np.array(l_2d).T.tolist()
print(l_2d_t)
print(type(l_2d_t))

#%%
df_t = pd.DataFrame(l_2d).T
print(df_t)
print(type(df_t))
l_2d_t = pd.DataFrame(l_2d).T.values.tolist()
print(l_2d_t)
print(type(l_2d_t))

#%%
l_2d_t_tuple = list(zip(*l_2d))
print(l_2d_t_tuple)
print(type(l_2d_t_tuple))
print(l_2d_t_tuple[0])
print(type(l_2d_t_tuple[0]))

#%%
l_2d_t = [list(x) for x in zip(*l_2d)]
print(l_2d_t)
print(type(l_2d_t))
print(l_2d_t[0])
print(type(l_2d_t[0]))

print(*l_2d)
print(list(zip([0, 1, 2], [3, 4, 5])))
print([list(x) for x in [(0, 3), (1, 4), (2, 5)]])