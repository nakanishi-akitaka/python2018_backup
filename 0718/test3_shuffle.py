# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 21:08:31 2018

@author: Akitaka
"""
import numpy as np
from sklearn.model_selection import ShuffleSplit, KFold
X = np.arange(10)
ss = ShuffleSplit(n_splits=10, test_size=0.1,
    random_state=0)
kf = KFold(n_splits=10, random_state=0, shuffle=True)

print('ShuffleSplit')
for train_index, test_index in ss.split(X):
    print("%s %s" % (train_index, test_index))
print()
print('KFold')
for train_index, test_index in kf.split(X):
    print("%s %s" % (train_index, test_index))

ss = ShuffleSplit(n_splits=10, test_size=0.1)
for i in range(10):
    print('ShuffleSplit')
    for train_index, test_index in ss.split(X):
        print("%s %s" % (train_index, test_index))
    print()        