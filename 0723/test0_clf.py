# -*- coding: utf-8 -*-
"""
Example of Machine Learning - Classification

Created on Mon Jul 23 10:58:27 2018

@author: Akitaka
"""

# ref
# ../0412/test8.py
# https://qiita.com/ishizakiiii/items/0650723cc2b4eef2c1cf
# ../0413/test5.py
# ../0416/test1.py
# ../0417/test1.py
# ../0413/test3.py
# ../0419/test2.py
#
# ref. follow of machine learning
# http://univprof.com/archives/16-02-11-2849465.html
# http://univprof.com/archives/16-05-01-2850729.html
# https://dev.classmethod.jp/machine-learning/introduction-scikit-learn/
# flow chart of choosing learning method
# http://scikit-learn.org/stable/tutorial/machine_learning_map/
# table of methods
# http://scikit-learn.org/stable/modules/classes.html
#
# Parameter estimation using grid search with cross-validation
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html

#
# import modules
#
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
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
#
# start of machine learning
#

# sample data
X, y = make_classification(n_samples=200, n_features=2, n_informative=2,
  n_redundant=0, n_classes=2, n_clusters_per_class=1)

X_train, X_test, y_train, y_test = \
 train_test_split(X, y, test_size=0.2)

#
# search range
 # https://datachemeng.com/supportvectorregression/
# https://datachemeng.com/fastoptsvrhyperparams/
# range_c = 2**np.arange( -5,  11, dtype=float)
# range_e = 2**np.arange( -10,  1, dtype=float)
# range_g = 2**np.arange( -20, 11, dtype=float)
# 1.gamma = miximize gram matrix
# 2.optimize only epsilon with C = 3 (when X is autoscaled) ang opted gamma
# 3.optimize only C with opted epsilon ang opted gamma
# 4.optimize only gamma with opted C and epsilon
range_c = 2**np.arange(  -3,  5, dtype=float)
range_g = 2**np.arange( -5,  0, dtype=float)
print()
print('Search range')
print('c = ', range_c[0], ' ... ',range_c[len(range_c)-1])
print('g = ', range_g[0], ' ... ',range_g[len(range_g)-1])
print()

# Set the parameters by cross-validation
param_grid = [
    {'kernel': ['rbf'], 'gamma': range_g,'C': range_c},
    ]
score = 'accuracy'
print("# Tuning hyper-parameters for {}".format(score))
print()

# ShuffleSplit or KFold(shuffle=True)
# http://scikit-learn.org/0.18/modules/cross_validation.html
# https://mail.google.com/mail/u/0/#sent/QgrcJHsbjCZNCXqKkMlpLbTXWjKWfzHljSl
# https://mail.google.com/mail/u/0/#sent/RdDgqcJHpWcvcDjPgjkjXHLgLnDfdlQzrnZXHZlrxmfB
#
# n_splits = 2, 5
# https://datachemeng.com/doublecrossvalidation/
# http://univprof.com/archives/16-06-12-3889388.html
# n_splits = 2, 5, 10
# https://datachemeng.com/modelvalidation/
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

clf = GridSearchCV(SVC(), param_grid, cv=cv, scoring='accuracy')
clf.fit(X_train, y_train)
print_gscv_score(clf)

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_test, y_pred))
print()

                            

# visualize 
# ref: https://pythondatascience.plavox.info/matplotlib/散布図
# http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
#
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors=['k', 'k', 'k'],
            linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=40, cmap=plt.cm.Paired,
            edgecolors='k')


