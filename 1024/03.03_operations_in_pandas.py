# -*- coding: utf-8 -*-
"""
https://jakevdp.github.io/PythonDataScienceHandbook/03.03-operations-in-pandas.html
Created on Wed Oct 24 13:59:00 2018

@author: Akitaka
"""

# Ufuncs: Index Preservation
import pandas as pd
import numpy as np
rng = np.random.RandomState(42)
ser = pd.Series(rng.randint(0, 10, 4))
print(ser)

df = pd.DataFrame(rng.randint(0, 10, (3, 4)),
                  columns=['A', 'B', 'C', 'D'])
print(df)

print(np.exp(ser))
print(np.sin(df * np.pi / 4))

#%%
# UFuncs: Index Alignment
## Index alignment in Series

area = pd.Series({'Alaska': 1723337, 'Texas': 695662,
                  'California': 423967}, name='area')
population = pd.Series({'California': 38332521, 'Texas': 26448193,
                        'New York': 19651127}, name='population')

print(population / area)

print(area.index | population.index)

#%%
A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])
print(A + B)
print(A.add(B, fill_value=0))

#%%
## Index alignment in DataFrame

A = pd.DataFrame(rng.randint(0, 20, (2, 2)),
                 columns=list('AB'))
print(A)

B = pd.DataFrame(rng.randint(0, 10, (3, 3)),
                 columns=list('BAC'))
print(B)

print(A + B)

fill = A.stack().mean()
print(A.add(B, fill_value=fill))

#%%
# Ufuncs: Operations Between DataFrame and Series
A = rng.randint(10, size=(3, 4))
print(A)
print(A - A[0])

df = pd.DataFrame(A, columns=list('QRST'))
print(df - df.iloc[0])
print(df.subtract(df['R'], axis=0))

halfrow = df.iloc[0, ::2]
print(halfrow)
print(df - halfrow)
