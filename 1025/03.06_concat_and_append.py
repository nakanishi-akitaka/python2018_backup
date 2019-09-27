# -*- coding: utf-8 -*-
"""
https://jakevdp.github.io/PythonDataScienceHandbook/03.06-concat-and-append.html
Created on Thu Oct 25 12:54:39 2018

@author: Akitaka
"""

import pandas as pd
import numpy as np
def make_df(cols, ind):
    data = {c: [str(c) + str(i) for i in ind] for c in cols}
    return pd.DataFrame(data, ind)

print(make_df('ABC', range(3)))
class display(object):
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
    
# Recall: Concatenation of NumPy Arrays
x = [1, 2, 3]
y = [4, 5, 6]
z = [7, 8, 9]
print(np.concatenate([x, y, z]))

x = [[1, 2],
     [3, 4]]
print(np.concatenate([x, x], axis=1))

#%%
# Simple Concatenation with pd.concat
ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])
print(ser1)
print(ser2)
print(pd.concat([ser1, ser2]))

#%%
df1 = make_df('AB', [1, 2])
df2 = make_df('AB', [3, 4])
display('df1', 'df2', 'pd.concat([df1, df2])')

#%%
df3 = make_df('AB', [0, 1])
df4 = make_df('CD', [0, 1])
display('df3', 'df4', "pd.concat([df3, df4], axis=1)")

#%%
## Duplicate indices
x = make_df('AB', [0, 1])
y = make_df('AB', [2, 3])
y.index = x.index  # make duplicate indices!
display('x', 'y', 'pd.concat([x, y])')

#%%
### Catching the repeats as an error
try:
    pd.concat([x, y], verify_integrity=True)
except ValueError as e:
    print("ValueError:", e)
    
### Ignoring the index
display('x', 'y', 'pd.concat([x, y], ignore_index=True)')

### Adding MultiIndex keys
display('x', 'y', "pd.concat([x, y], keys=['x', 'y'])")

#%%
## Concatenation with joins
df5 = make_df('ABC', [1, 2])
df6 = make_df('BCD', [3, 4])
display('df5', 'df6', 'pd.concat([df5, df6])')

display('df5', 'df6',
        "pd.concat([df5, df6], join='inner')")

## The append() method
display('df1', 'df2', 'df1.append(df2)')