# -*- coding: utf-8 -*-
"""
https://jakevdp.github.io/PythonDataScienceHandbook/03.04-missing-values.html
Created on Wed Oct 24 14:12:42 2018

@author: Akitaka
"""
# Trade-Offs in Missing Data Conventions
# Missing Data in Pandas
## None:Pythonic missing data
#%%
import numpy as np
import pandas as pd
vals1 = np.array([1, None, 3, 4])
print(vals1)

#%%
## Nan:Missing numerical data
vals2 = np.array([1, np.nan, 3, 4])
print(vals2.dtype)
print(1 + np.nan)
print(0 * np.nan)
print(vals2.sum(), vals2.min(), vals2.max())
print(np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2))

#%%
## NaN and None in Pandas
print(pd.Series([1, np.nan, 2, None]))
x = pd.Series(range(2), dtype=int)
print(x)
x[0] = None
print(x)

#%%
# Operating on Null Values
## Detecting null values
data = pd.Series([1, np.nan, 'hello', None])
print(data.isnull())
print(data[data.notnull()])

#%%
## Dropping null values

print(data.dropna())
df = pd.DataFrame([[1,      np.nan, 2],
                   [2,      3,      5],
                   [np.nan, 4,      6]])
print(df)
print(df.dropna())
print(df.dropna(axis='columns'))

#df.to_csv("test.csv")
#df = pd.read_csv("test.csv", index_col=0)
#print(df.dropna())

df[3] = np.nan
print(df)
print(df.dropna(axis='columns', how='all'))
print(df.dropna(axis='rows', thresh=3))

#%%
## Filling null values
data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
print(data)
print(data.fillna(0))
print(data.fillna(method='ffill'))
print(data.fillna(method='bfill'))

print(df)
print(df.fillna(method='ffill', axis=1))

