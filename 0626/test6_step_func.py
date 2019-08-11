# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 15:02:39 2018

@author: Akitaka
"""


import numpy as np
import pandas as pd
# print(np.random.randint(0, 2, 10))
# x = np.random.randint(0, 2, 10)
x = np.arange(-5,5,1)
y = 1 * (x > 0) + 1 
print(x)
print(y)

# read data from csv file
name = '../2018/0625/test1_cls.csv'
data = np.array(pd.read_csv(name))[:,:]
y=data[:,8]
X=data[:,0:2]
print(y)
y1=y

name = '../20180625/test1.csv'
data = np.array(pd.read_csv(name))[:,:]
y=data[:,8]
X=data[:,0:2]
print(y)
y2 = 1.0 * (y > 0) + 1.0 
print(y2)
print(y1-y2)
