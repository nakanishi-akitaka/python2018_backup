# -*- coding: utf-8 -*-
"""
Example of Machine Learning - Classification

Created on Thu Jul 26 09:19:06 2018

@author: Akitaka
"""

import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
from time                    import time
from sklearn.datasets        import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics         import classification_report
from sklearn.metrics         import confusion_matrix
from sklearn.svm             import SVC
from my_library              import print_gscv_score

print(__doc__)

start = time()

X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
  n_redundant=0, n_classes=2, n_clusters_per_class=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

mod = SVC() 

#
# search range
# range_c = 2**np.arange( -5,  11, dtype=float)
# range_g = 2**np.arange( -20, 11, dtype=float)
range_c = 2**np.arange(  -3,  5, dtype=float)
range_g = 2**np.arange( -5,  0, dtype=float)
print()
print('Search range')
print('c = ', range_c[0], ' ... ',range_c[len(range_c)-1])
print('g = ', range_g[0], ' ... ',range_g[len(range_g)-1])
print()

param_grid = [
    {'kernel': ['rbf'], 'gamma': range_g,'C': range_c},
    ]
score = 'accuracy'
print("# Tuning hyper-parameters for {}".format(score))
print()

cv = ShuffleSplit(n_splits=5, test_size=0.2)
gscv = GridSearchCV(mod, param_grid, cv=cv, scoring='accuracy')
gscv.fit(X_train, y_train)
print_gscv_score(gscv)

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, gscv.predict(X_test)
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
print()