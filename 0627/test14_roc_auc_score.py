# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 20:55:58 2018

@author: Akitaka
"""

# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
import numpy as np
from sklearn.metrics import roc_auc_score
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
print(roc_auc_score(y_true, y_scores))

# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
from sklearn.metrics import average_precision_score
print(average_precision_score(y_true, y_scores))

# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
print(fpr)
print(tpr)
print(thresholds)

# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
from sklearn.metrics import auc
print(auc(fpr, tpr))

# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
print(precision)
print(recall)
print(thresholds)
