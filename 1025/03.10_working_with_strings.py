# -*- coding: utf-8 -*-
"""
https://jakevdp.github.io/PythonDataScienceHandbook/03.10-working-with-strings.html
Created on Thu Oct 25 14:42:54 2018

@author: Akitaka
"""
#%%
import numpy as np
x = np.array([2, 3, 5, 7, 11, 13])
x * 2

data = ['peter', 'Paul', 'MARY', 'gUIDO']
[s.capitalize() for s in data]

data = ['peter', 'Paul', None, 'MARY', 'gUIDO']
import pandas as pd
names = pd.Series(data)
names
names.str.capitalize()

#%%
# Tables of Pandas String Methods
monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
                   'Eric Idle', 'Terry Jones', 'Michael Palin'])

## Methods similar to Python string methods
monte.str.lower()
monte.str.len()
monte.str.startswith('T')
monte.str.split()

## Methods using regular expressions
monte.str.extract('([A-Za-z]+)', expand=False)
monte.str.findall(r'^[^AEIOU].*[^aeiou]$')

## Miscellaneous methods
### Vectorized item access and slicing
monte.str[0:3]
monte.str.split().str.get(-1)

### Indicator variables
full_monte = pd.DataFrame({'name': monte,
                           'info': ['B|C|D', 'B|D', 'A|C',
                                    'B|D', 'B|C', 'B|C|D']})
full_monte
full_monte['info'].str.get_dummies('|')








