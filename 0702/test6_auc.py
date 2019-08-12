# -*- coding: utf-8 -*-
"""
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html

Created on Mon Jul  2 13:11:39 2018

@author: Akitaka
"""
import numpy as np
from sklearn import metrics
y = np.array([1, 1, 2, 2])
pred = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
print(metrics.auc(fpr, tpr))

y = np.array([0, 0, 1, 1])
pred = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, pred)
print(metrics.auc(fpr, tpr))
