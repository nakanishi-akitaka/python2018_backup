# -*- coding: utf-8 -*-
"""
https://jakevdp.github.io/PythonDataScienceHandbook/03.08-aggregation-and-grouping.html
Created on Thu Oct 25 13:30:28 2018

@author: Akitaka
"""
#%%
import numpy as np
import pandas as pd

class display(object):
    """Display HTML representation of multiple objects"""
    template = """<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
    </div>"""
    def __init__(self, *args):
        self.args = args
        
    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                         for a in self.args)
    
    def __repr__(self):
        return '\n\n'.join(a + '\n' + repr(eval(a))
                           for a in self.args)

# Planets Data
import seaborn as sns
planets = sns.load_dataset('planets')
planets.shape
planets.head()

#%%
# Simple Aggregation in Pandas
rng = np.random.RandomState(42)
ser = pd.Series(rng.rand(5))
print(ser)
print(ser.sum())
print(ser.mean())

df = pd.DataFrame({'A': rng.rand(5),
                   'B': rng.rand(5)})
print(df)
print(df.mean())
print(df.mean(axis='columns'))

print(planets.dropna().describe())

#%%
# GroupBy: Split, Apply, Combine
## Split, apply, combine
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data': range(6)}, columns=['key', 'data'])
print(df)
print(df.groupby('key'))
print(df.groupby('key').sum())

#%%
## The GroupBy object
### Column indexing
print(planets.groupby('method'))
print(planets.groupby('method')['orbital_period'])
print(planets.groupby('method')['orbital_period'].median())

#%%
### Iteration over groups
for (method, group) in planets.groupby('method'):
    print("{0:30s} shape={1}".format(method,group.shape))

### Dispatch methods
print(planets.groupby('method')['year'].describe().unstack())

## Aggregate, filter, transform, apply
rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data1': range(6),
                   'data2': rng.randint(0, 10, 6)},
                   columns = ['key', 'data1', 'data2'])
print(df)

### Aggregation
print(df.groupby('key').aggregate(['min', np.median, max]))

print(df.groupby('key').aggregate({'data1':'min','data2':'max'}))
#%%
### Filtering
def filter_func(x):
    return x['data2'].std() > 4

print(df.groupby('key').filter(filter_func))

### Transformation
df.groupby('key').transform(lambda x: x - x.mean())

### The apply() method
def norm_by_data2(x):
    x['data1'] /= x['data2'].sum()
    return x

print(df.groupby('key'))
print(df.groupby('key').apply(norm_by_data2))

#%%
## Specifying the split key
L = [0, 1, 0, 1, 2, 0]
print(df.groupby(L).sum())
print(df.groupby(df['key']).sum())

#%%
### A dictionary or series mapping index to group
df2 = df.set_index('key')
mapping = {'A': 'vowel', 'B': 'consonant', 'C': 'consonant'}
print(df2.groupby(mapping).sum())

#%%
### Any Python function
print(df2.groupby(str.lower).mean())

#%%
### A list of valid keys
print(df2.groupby([str.lower, mapping]).mean())

#%%
## Grouping example
decade = 10 * (planets['year'] // 10)
decade = decade.astype(str) + 's'
decade.name = 'decade'
print(planets.groupby(['method', decade])['number'].sum().unstack().fillna(0))
