# -*- coding: utf-8 -*-
"""
https://jakevdp.github.io/PythonDataScienceHandbook/02.09-structured-data-numpy.html
Created on Mon Oct 22 21:17:18 2018

@author: Akitaka
"""

import numpy as np
name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]

x = np.zeros(4, dtype=int)

data = np.zeros(4, dtype={'names':('name', 'age', 'weight'),
                          'formats':('U10', 'i4', 'f8')})
print(data.dtype)

data['name'] = name
data['age'] = age
data['weight'] = weight

#%%
print(data)
print(data['name'])
print(data[0])
print(data[-1]['name'])
print(data[data['age'] < 30]['name'])


#%%
# Creating Structured Arrays
print(np.dtype({'names':('name', 'age', 'weight'),
                'formats':('U10', 'i4', 'f8')}))
print(np.dtype({'names':('name', 'age', 'weight'),
                'formats':((np.str_, 10), int, np.float32)}))

print(np.dtype([('name', 'S10'), ('age', 'i4'), ('weight', 'f8')]))
print(np.dtype('S10,i4,f8'))


#%%
# More Advanced Compound Types
tp = np.dtype([('id', 'i8'), ('mat', 'f8', (3, 3))])
X = np.zeros(1, dtype=tp)
print(X[0])
print(X['mat'][0])

#%%
# RecordArrays: Structured Arrays with a Twist
print(data['age'])

data_rec = data.view(np.recarray)
print(data_rec.age)

