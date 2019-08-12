# -*- coding: utf-8 -*-
"""
http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
Created on Tue Jul  3 12:39:48 2018

@author: Akitaka
"""
print(__doc__)

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

# #############################################################################
# Data IO and generation

# Import some data to play with
iris = datasets.load_iris()
X = iris.data
y = iris.target
X, y = X[y != 2], y[y != 2]
n_samples, n_features = X.shape

# Add noisy features
random_state = np.random.RandomState(0)
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# #############################################################################
# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=6)
classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=random_state)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
i = 0
for train, test in cv.split(X, y):
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    print(probas_[:,1])
    decision_ = classifier.fit(X[train], y[train]).decision_function(X[test])
    print(decision_)
    print(X[test].shape)
    print(X[train].shape)
    
W = np.array([[1,2,3],[4,5,6]])
print(W.shape)