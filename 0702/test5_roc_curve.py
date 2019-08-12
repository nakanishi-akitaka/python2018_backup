# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 13:09:39 2018

@author: Akitaka
"""

import numpy as np
from sklearn import metrics
y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)

print(fpr)
print(tpr)
print(thresholds)

y = np.array([0, 0, 1, 1])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores)

print(fpr)
print(tpr)
print(thresholds)
