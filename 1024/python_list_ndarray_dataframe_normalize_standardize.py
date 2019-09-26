# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-list-ndarray-dataframe-normalize-standardize/
Created on Wed Oct 24 12:15:33 2018

@author: Akitaka
"""

import scipy.stats
l = [0, 1, 2, 3, 4]
print(l)

print(scipy.stats.zscore(l))
print(type(scipy.stats.zscore(l)))

print(scipy.stats.zscore(l, ddof=-1))

#%%
l_2d = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
print(l_2d)
print(scipy.stats.zscore(l_2d))
print(scipy.stats.zscore(l_2d, ddof=1))

print(scipy.stats.zscore(l_2d, axis=1))
print(scipy.stats.zscore(l_2d, axis=1, ddof=1))

print(scipy.stats.zscore(l_2d, axis=None))
print(scipy.stats.zscore(l_2d, axis=None, ddof=1))

#%%
from sklearn import preprocessing
l = [0, 1, 2, 3, 4]
print(l)
l_2d = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
print(l_2d)

mm = preprocessing.MinMaxScaler()
l_2d_min_max = mm.fit_transform(l_2d)
print(l_2d_min_max)
print(type(l_2d_min_max))

#%%
print(preprocessing.minmax_scale(l))
print(preprocessing.minmax_scale(l_2d))
print(preprocessing.minmax_scale(l_2d, axis=1))

#%%
ss = preprocessing.StandardScaler()
l_2d_standardization = ss.fit_transform(l_2d)
print(l_2d_standardization)
print(type(l_2d_standardization))
print(preprocessing.scale(l))
print(preprocessing.scale(l_2d))
print(preprocessing.scale(l_2d, axis=1))

#%%

import statistics
import pprint
l = [0, 1, 2, 3, 4]
print(l)
def min_max(l):
    l_min = min(l)
    l_max = max(l)
    return [(i - l_min) / (l_max - l_min) for i in l]

print(min_max(l))

def standardization(l):
    l_mean = statistics.mean(l)
    l_stdev = statistics.stdev(l)
    return [(i - l_mean) / l_stdev for i in l]

pprint.pprint(standardization(l))

def standardization_p(l):
    l_mean = statistics.mean(l)
    l_pstdev = statistics.pstdev(l)
    return [(i - l_mean) / l_pstdev for i in l]

pprint.pprint(standardization_p(l))

#%%
l_2d = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
pprint.pprint([min_max(l_1d) for l_1d in l_2d], width=40)
pprint.pprint([standardization(l_1d) for l_1d in l_2d], width=40)
pprint.pprint([standardization_p(l_1d) for l_1d in l_2d])

#%%
l_2d_min_max_col = list(zip(*[min_max(l_1d) for l_1d in list(zip(*l_2d))]))
pprint.pprint(l_2d_min_max_col, width=40)

l_2d_standardization_col = list(zip(*[standardization(l_1d) for l_1d in list(zip(*l_2d))]))
pprint.pprint(l_2d_standardization_col, width=40)

l_2d_standardization_p_col = list(zip(*[standardization_p(l_1d) for l_1d in list(zip(*l_2d))]))
pprint.pprint(l_2d_standardization_p_col)

#%%
def min_max_2d_all(l_2d):
    l_flatten = sum(l_2d, [])
    l_2d_min = min(l_flatten)
    l_2d_max = max(l_flatten)
    return [[(i - l_2d_min) / (l_2d_max - l_2d_min) for i in l_1d] for l_1d in l_2d]

pprint.pprint(min_max_2d_all(l_2d), width=40)

def standardization_2d_all(l):
    l_flatten = sum(l_2d, [])
    l_2d_mean = statistics.mean(l_flatten)
    l_2d_stdev = statistics.stdev(l_flatten)
    return [[(i - l_2d_mean) / l_2d_stdev for i in l_1d]
            for l_1d in l_2d]

pprint.pprint(standardization_2d_all(l_2d))

def standardization_p_2d_all(l):
    l_flatten = sum(l_2d, [])
    l_2d_mean = statistics.mean(l_flatten)
    l_2d_pstdev = statistics.pstdev(l_flatten)
    return [[(i - l_2d_mean) / l_2d_pstdev for i in l_1d]
            for l_1d in l_2d]

pprint.pprint(standardization_p_2d_all(l_2d))


#%%
import numpy as np
import scipy.stats
from sklearn import preprocessing
a = np.array([0, 1, 2, 3, 4])
print(a)
print((a - a.min()) / (a.max() - a.min()))
print((a - a.mean()) / a.std())
print((a - a.mean()) / a.std(ddof=1))

#%%
a_2d = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print(a_2d)

def min_max(x, axis=None):
    x_min = x.min(axis=axis, keepdims=True)
    x_max = x.max(axis=axis, keepdims=True)
    return (x - x_min) / (x_max - x_min)

print(min_max(a_2d))
print(min_max(a_2d, axis=0))
print(min_max(a_2d, axis=1))
print(min_max(a))

#%%
def standardization(x, axis=None, ddof=0):
    x_mean = x.mean(axis=axis, keepdims=True)
    x_std = x.std(axis=axis, keepdims=True, ddof=ddof)
    return (x - x_mean) / x_std

print(standardization(a_2d))
print(standardization(a_2d, ddof=1))
print(standardization(a_2d, axis=0))
print(standardization(a_2d, axis=0, ddof=1))
print(standardization(a_2d, axis=1))
print(standardization(a_2d, axis=1, ddof=1))
print(standardization(a))
print(standardization(a, ddof=1))

#%%
print(scipy.stats.zscore(a))
print(scipy.stats.zscore(a_2d))
print(scipy.stats.zscore(a_2d, axis=None, ddof=1))

#%%
mm = preprocessing.MinMaxScaler()
print(mm.fit_transform(a_2d.astype(float)))
print(preprocessing.minmax_scale(a.astype(float)))
print(preprocessing.minmax_scale(a_2d.astype(float), axis=1))

ss = preprocessing.StandardScaler()
print(ss.fit_transform(a_2d.astype(float)))
print(preprocessing.scale(a.astype(float)))
print(preprocessing.scale(a_2d.astype(float), axis=1))

#%%
import pandas as pd
import scipy.stats
from sklearn import preprocessing
df = pd.DataFrame([[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  columns=['col1', 'col2', 'col3'],
                  index=['a', 'b', 'c'])

print(df)
print((df - df.min()) / (df.max() - df.min()))
print(((df.T - df.T.min()) / (df.T.max() - df.T.min())).T)
print((df - df.values.min()) / (df.values.max() - df.values.min()))

#%%
print((df - df.mean()) / df.std())
print((df - df.mean()) / df.std(ddof=0))
print(((df.T - df.T.mean()) / df.T.std()).T)
print(((df.T - df.T.mean()) / df.T.std(ddof=0)).T)
print((df - df.values.mean()) / df.values.std())
print((df - df.values.mean()) / df.values.std(ddof=1))

df_ = df.copy()
s = df_['col1']
df_['col1_min_max'] = (s - s.min()) / (s.max() - s.min())
df_['col1_standardization'] = (s - s.mean()) / s.std()

print(df_)

#%%
print(scipy.stats.zscore(df))
print(type(scipy.stats.zscore(df)))
print(scipy.stats.zscore(df, axis=None, ddof=1))

df_standardization = pd.DataFrame(scipy.stats.zscore(df),
                                  index=df.index, columns=df.columns)
print(df_standardization)

df_ = df.copy()
df_['col1_standardization'] = scipy.stats.zscore(df_['col1'])
print(df_)

#%%
mm = preprocessing.MinMaxScaler()
print(mm.fit_transform(df))
print(type(mm.fit_transform(df)))
print(preprocessing.minmax_scale(df))
print(type(preprocessing.minmax_scale(df)))

df_min_max = pd.DataFrame(mm.fit_transform(df),
                          index=df.index, columns=df.columns)

print(df_min_max)

df_ = df.copy()
s = df_['col1'].astype(float)
df_['col1_min_max'] = preprocessing.minmax_scale(s)
df_['col1_standardization'] = preprocessing.scale(s)

print(df_)
print(df.describe())
print(df.T.describe())
