# -*- coding: utf-8 -*-
"""
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
Created on Mon Jul  2 13:22:48 2018

@author: Akitaka
"""

import numpy as np
from sklearn.metrics import average_precision_score
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
print(average_precision_score(y_true, y_scores))
