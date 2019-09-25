# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 15:33:49 2018

@author: Akitaka
"""

# The Pandas Series Object
import numpy as np
import pandas as pd
data = pd.Series([0.25, 0.5, 0.75, 1.0])
print(data)
print(data.values)
print(data.index)
print(data[1])
print(data[1:3])

#%%
## Series as generalized NumPy array
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
print(data)
print(data['b'])

data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=[2, 5, 3, 7])
print(data)
print(data[5])

#%%
## Series as specialized dictionary
population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
print(population)
print(population['California'])
print(population['California':'Illinois'])

#%%
## Constructing Series objects
pd.Series([2, 4, 6])
pd.Series(5, index=[100, 200, 300])
pd.Series({2:'a', 1:'b', 3:'c'})
pd.Series({2:'a', 1:'b', 3:'c'}, index=[3, 2])

#%%
# The Pandas DataFrame Object
## DataFrame as a generalized NumPy array
area_dict = {'California': 423967, 'Texas': 695662, 'New York': 141297,
             'Florida': 170312, 'Illinois': 149995}
area = pd.Series(area_dict)
print(area)

states = pd.DataFrame({'population': population,
                       'area': area})
print(states)
print(states.index)
print(states.columns)

#%%
## DataFrame as specialized dictionary
print(states['area'])


## Constructing DataFrame objects 
### From a single Series object
pd.DataFrame(population, columns=['population'])

### From a list of dicts
data = [{'a': i, 'b': 2 * i}
        for i in range(3)]
pd.DataFrame(data)

pd.DataFrame([{'a': 1, 'b': 2}, {'b': 3, 'c': 4}])

### From a dictionary of Series objects
pd.DataFrame({'population': population,
              'area': area})

### From a two-dimensional NumPy array
pd.DataFrame(np.random.rand(3, 2),
             columns=['foo', 'bar'],
             index=['a', 'b', 'c'])

### From a NumPy structured array
A = np.zeros(3, dtype=[('A', 'i8'), ('B', 'f8')])
print(A)
pd.DataFrame(A)

#%%
# The Pandas Index Object
ind = pd.Index([2, 3, 5, 7, 11])
ind

## Index as immutable array
ind[1]
ind[::2]
print(ind.size, ind.shape, ind.ndim, ind.dtype)

## Index as ordered set
indA = pd.Index([1, 3, 5, 7, 9])
indB = pd.Index([2, 3, 5, 7, 11])
indA & indB
indA | indB
indA ^ indB
