# -*- coding: utf-8 -*-
"""
https://jakevdp.github.io/PythonDataScienceHandbook/03.02-data-indexing-and-selection.html
Created on Tue Oct 23 15:48:53 2018

@author: Akitaka
"""

# Data Selection in Series
## Series as dictionary
import pandas as pd
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
print(data)
print(data['b'])
print('a' in data)
print(data.keys)
print(list(data.items()))
data['e'] = 1.25
print(data)

#%%
## Series as one-dimensional array
print(data['a':'c'])
print(data[0:2])
print(data[(data > 0.3) & (data < 0.8)])
print(data[['a', 'e']])


#%%
## Indexers: loc, iloc, and ix
data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])
print(data)
print(data[1])
print(data[1:3])
print(data.loc[1])
print(data.loc[1:3])
print(data.iloc[1])
print(data.iloc[1:3])

#%%
# Data Selection in DataFrame
## DataFrame as a dictionary

area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': 19552860,
                 'Illinois': 12882135})
data = pd.DataFrame({'area':area, 'pop':pop})
print(data)
print(data['area'])
print(data.area)
print(data.area is data['area'])
print(data.pop is data['pop'])

data['density'] = data['pop'] / data['area']
print(data)

#%%
## DataFrame as two-dimensional array
print(data.values)
print(data.T)
print(data.values[0])
print(data['area'])

print(data.iloc[:3,:2])
print(data.loc[:'Illinois',:'pop'])
# print(data.ix[:3, :'pop'])
print(data.loc[data.density > 100, ['pop', 'density']])
data.iloc[0, 2] = 90
print(data)

#%%
## Additional indexing conventions
print(data['Florida':'Illinois'])
print(data[1:3])
print(data[data.density > 100])



