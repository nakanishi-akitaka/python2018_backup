# -*- coding: utf-8 -*-
"""
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve

Created on Mon Jul  2 13:17:44 2018

@author: Akitaka
"""

import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
print(precision)
print(recall)
print(thresholds)
print(average_precision_score(y_true, y_scores))

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
print(fpr)
print(tpr)
print(thresholds)
print(auc(fpr, tpr))

