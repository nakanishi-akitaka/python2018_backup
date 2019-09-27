# -*- coding: utf-8 -*-
"""
https://jakevdp.github.io/PythonDataScienceHandbook/03.07-merge-and-join.html
Created on Thu Oct 25 13:15:13 2018

@author: Akitaka
"""
#%%
import pandas as pd
import numpy as np

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
        
# Categories of Joins
## One-to-one joins
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})
display('df1', 'df2')

df3 = pd.merge(df1, df2)
df3

#%%
## Many-to-one joins
df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                    'supervisor': ['Carly', 'Guido', 'Steve']})
display('df3', 'df4', 'pd.merge(df3, df4)')

#%%
## Many-to-many joins
df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
                              'Engineering', 'Engineering', 'HR', 'HR'],
                    'skills': ['math', 'spreadsheets', 'coding', 'linux',
                               'spreadsheets', 'organization']})
display('df1', 'df5', "pd.merge(df1, df5)")
print(pd.merge(df1, df5))

#%%
# Specification of the Merge Key
## The on keyword
print(pd.merge(df1, df2, on='employee'))

#%%
## The left_on and right_on keywords
df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'salary': [70000, 80000, 120000, 90000]})
print(pd.merge(df1, df3, left_on="employee", right_on="name"))

print(pd.merge(df1, df3, left_on="employee", right_on="name").drop('name', axis=1))

#%%
## The left_index and right_index keywords
df1a = df1.set_index('employee')
df2a = df2.set_index('employee')
print(pd.merge(df1a, df2a, left_index=True, right_index=True))

print(pd.merge(df1a, df3, left_index=True, right_on='name'))

#%%
# Specifying Set Arithmetic for Joins


df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
                    'food': ['fish', 'beans', 'bread']},
                   columns=['name', 'food'])
df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
                    'drink': ['wine', 'beer']},
                   columns=['name', 'drink'])

print(pd.merge(df6, df7))
print(pd.merge(df6, df7, how='inner'))
print(pd.merge(df6, df7, how='outer'))
print(pd.merge(df6, df7, how='left'))
print(pd.merge(df6, df7, how='right'))

#%%
# Overlapping Column Names: The suffixes Keyword
df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [1, 2, 3, 4]})
df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [3, 1, 4, 2]})
print(pd.merge(df8, df9, on="name"))
print(pd.merge(df8, df9, on="name", suffixes=["_L", "_R"]))


