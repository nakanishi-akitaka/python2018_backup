# -*- coding: utf-8 -*-
"""
sklearn.metrics.roc_auc_score
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

Created on Mon Jul  2 13:02:41 2018

@author: Akitaka
"""
import numpy as np
from sklearn.metrics import roc_auc_score
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
print(roc_auc_score(y_true, y_scores))
