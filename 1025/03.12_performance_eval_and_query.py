# -*- coding: utf-8 -*-
"""
https://jakevdp.github.io/PythonDataScienceHandbook/03.12-performance-eval-and-query.html
Created on Thu Oct 25 15:03:18 2018

@author: Akitaka
"""
#%%
# Motivating query() and eval(): Compound Expressions
import numpy as np
rng = np.random.RandomState(42)
x = rng.rand(10)
y = rng.rand(10)
mask = (x > 0.5) & (y < 0.5)
tmp1 = (x > 0.5)
tmp2 = (x < 0.5)
mask = tmp1 & tmp2

import numexpr
mask_numexpr = numexpr.evaluate('(x > 0.5) & (y < 0.5)')
np.allclose(mask, mask_numexpr)

#%%
# pandas.eval() for Efficient Operations
import pandas as pd
nrows, ncols = 100000, 100
rng = np.random.RandomState(42)
df1, df2, df3, df4 = (pd.DataFrame(rng.rand(nrows, ncols)) for i in range(4))
np.allclose(df1 + df2 + df3  + df4, pd.eval('df1 + df2 + df3 + df4'))

#%%
## Operations supported by pd.eval()
df1, df2, df3, df4, df5 = (pd.DataFrame(rng.randint(0, 1000, (100, 3))) for i in range(5))

### Arithmetic operators
result1 = -df1 * df2 / (df3 + df4) - df5
result2 = pd.eval('-df1 * df2 / (df3 + df4) - df5')
np.allclose(result1, result2)

#%%
### Comparison operators
result1 = (df1 < df2) & (df2 <= df3) & (df3 != df4)
result2 = pd.eval('df1 < df2 <= df3 != df4')
np.allclose(result1, result2)

#%%
### Bitwise operators
result1 = (df1 < 0.5) & (df2 < 0.5) | (df3 < df4)
result2 = pd.eval('(df1 < 0.5) & (df2 < 0.5) | (df3 < df4)')
np.allclose(result1, result2)
result3 = pd.eval('(df1 < 0.5) and (df2 < 0.5) or (df3 < df4)')
np.allclose(result1, result3)

#%%
### Object attributes and indices
results1 = df2.T[0] + df3.iloc[1]
results2 = pd.eval('df2.T[0] + df3.iloc[1]')
np.allclose(result1, result2)

#%%
# DataFrame.eval() for Column-Wise Operations
df = pd.DataFrame(rng.rand(1000, 3), columns=['A', 'B', 'C'])
df.head()

#%%
result1 = (df['A'] + df['B']) / (df['C'] - 1)
result2 = pd.eval("(df.A + df.B) / (df.C - 1)")
np.allclose(result1, result2)

result3 = df.eval('(A + B) / (C - 1)')
np.allclose(result1, result3)

## Assignment in DataFrame.eval()
df.head()
df.eval('D = (A + B) / C', inplace=True)
df.head()
df.eval('D = (A - B) / C', inplace=True)
df.head()

#%%
## Local variables in DataFrame.eval()
column_mean = df.mean(1)
result1 = df['A'] + column_mean
result2 = df.eval('A + @column_mean')
np.allclose(result1, result2)

# DataFrame.query() Method
result1 = df[(df.A < 0.5) & (df.B < 0.5)]
result2 = pd.eval('df[(df.A < 0.5) & (df.B < 0.5)]')
np.allclose(result1, result2)

result3 = df.query('A < 0.5 and B < 0.5')
np.allclose(result1, result3)

Cmean= df['C'].mean()
result1 = df[(df.A < Cmean) & (df.B < Cmean)]
result2 = df.query('A < @Cmean and B < @Cmean')
np.allclose(result1, result2)

# Performance: When to Use These Functions
x = df[(df.A < 0.5) & (df.B < 0.5)]
tmp1 = df.A < 0.5
tmp2 = df.B < 0.5
tmp3 = tmp1 & tmp2
x = df[tmp3]
df.values.nbytes
