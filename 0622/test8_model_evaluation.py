# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 16:16:55 2018

@author: Akitaka
"""

# http://scikit-learn.org/stable/modules/model_evaluation.html
# 3.3.2.4. Confusion matrix

from sklearn import metrics

y_true = [0, 0, 0, 1, 1, 1, 1, 1]
y_pred = [0, 1, 0, 1, 0, 1, 0, 1]
tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
print(metrics.confusion_matrix(y_true, y_pred))
print(metrics.confusion_matrix(y_true, y_pred).ravel())

print('True Positive, False Positive, False Negative, True Positive')
print(tn, fp, fn, tp)
print('False Positive Rate, True Positive Rate')
print(fp/(tn+fp), tp/(fn+tp))

# http://scikit-learn.org/stable/modules/model_evaluation.html
# 3.3.2.12. Receiver operating characteristic (ROC)
import numpy as np
from sklearn.metrics import roc_curve
y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = roc_curve(y, scores, pos_label=2)
print(fpr)
print(tpr)
print(thresholds)

