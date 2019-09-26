# -*- coding: utf-8 -*-
"""
https://jakevdp.github.io/PythonDataScienceHandbook/03.05-hierarchical-indexing.html
Created on Wed Oct 24 14:38:20 2018

@author: Akitaka
"""
#%%
import pandas as pd
import numpy as np

# A Multiply Indexed Series
## The bad way
index = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,
               18976457, 19378102,
               20851820, 25145561]
pop = pd.Series(populations, index=index)
print(pop)
print(pop[('California', 2010):('Texas', 2000)])
print(pop[[i for i in pop.index if i[1] == 2010]])

#%%
## The Better Way: Pandas MultiIndex
index = pd.MultiIndex.from_tuples(index)
print(index)

pop = pop.reindex(index)
print(pop)
print(pop[:,2010])

## MultiIndex as extra dimension
pop_df = pop.unstack()
print(pop_df)

print(pop_df.stack())
pop_df = pd.DataFrame({'total': pop,
                       'under18': [9267089, 9284094,
                                   4687374, 4318033,
                                   5906301, 6879014]})
print(pop_df)
f_u18 = pop_df['under18'] / pop_df['total']
print(f_u18.unstack())

#%%
# Methods of MultiIndex Creation
df = pd.DataFrame(np.random.rand(4, 2),
                  index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                  columns=['data1', 'data2'])
print(df)
data = {('California', 2000): 33871648,
        ('California', 2010): 37253956,
        ('Texas', 2000): 20851820,
        ('Texas', 2010): 25145561,
        ('New York', 2000): 18976457,
        ('New York', 2010): 19378102}
print(pd.Series(data))

#%%
## Explicit MultiIndex constructors
print(pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]]))
print(pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)]))
print(pd.MultiIndex.from_product([['a', 'b'], [1, 2]]))
print(pd.MultiIndex(levels=[['a', 'b'], [1, 2]],
              labels=[[0, 0, 1, 1], [0, 1, 0, 1]]))

#%%
## MultiIndex level names
pop.index.names = ['state', 'year']
print(pop)

#%%
## MultiIndex for columns

index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                   names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
                                     names=['subject', 'type'])

data = np.round(np.random.randn(4 ,6), 1)
data[:, ::2] *= 10
data += 37
health_data = pd.DataFrame(data, index=index, columns=columns)
print(health_data)
print(health_data['Guido'])

#%%
# Indexing and Slicing a MultiIndex
## Multiply indexed Series
print(pop)
print(pop['California', 2000])
print(pop['California'])
print(pop['California':'New York'])
print(pop[:, 2000])
print(pop[pop > 22000000])
print(pop['California', 'Texas'])

#%%
## Multiply indexed DataFrames
print(health_data)
print(health_data['Guido', 'HR'])
print(health_data.iloc[:2, :2])
print(health_data.loc[:, ('Bob', 'HR')])
idx = pd.IndexSlice
print(health_data.loc[idx[:, 1], idx[:, 'HR']])

#%%
# Rearranging Multi-Indices
## Sorted and unsorted indices
index = pd.MultiIndex.from_product([['a', 'c', 'b'], [1, 2]])
data = pd.Series(np.random.rand(6), index=index)
data.index.names = ['char', 'int']
print(data)

data = data.sort_index()
print(data)
print(data['a':'b'])

#%%
## Stacking and unstacking indices
print(pop.unstack(level=0))
print(pop.unstack(level=1))
print(pop.unstack().stack())

#%%
## Index setting and resetting
pop_flat = pop.reset_index(name='population')
print(pop_flat)
print(pop_flat.set_index(['state', 'year']))

#%%
# Data Aggregations on Multi-Indices
print(health_data)
data_mean = health_data.mean(level='year')
print(data_mean)
print(data_mean.mean(axis=1, level='type'))
