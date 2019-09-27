# -*- coding: utf-8 -*-
"""
https://jakevdp.github.io/PythonDataScienceHandbook/03.09-pivot-tables.html
Created on Thu Oct 25 14:31:26 2018

@author: Akitaka
"""
#%%
import numpy as np
import pandas as pd
import seaborn as sns
titanic = sns.load_dataset('titanic')
titanic.head()
titanic.groupby('sex')[['survived']].mean()
titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack()

#%%
# Pivot Table Syntax
titanic.pivot_table('survived', index='sex', columns='class')

## Multi-level pivot tables
age = pd.cut(titanic['age'], [0, 18, 80])
titanic.pivot_table('survived', ['sex', age], 'class')

fare = pd.qcut(titanic['fare'], 2)
titanic.pivot_table('survived', ['sex', age], [fare, 'class'])

## Additional pivot table options
titanic.pivot_table(index='sex', columns='class',
                    aggfunc={'survived':sum, 'fare':'mean'})

titanic.pivot_table('survived', index='sex', columns='class', margins=True)