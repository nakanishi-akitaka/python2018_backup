# -*- coding: utf-8 -*-
"""
  Double Cross Validation
+ Support Vector Machine
+ One-Class Support Vector Machine 
  Classification
  
Created on Wed Jul 25 15:36:22 2018

@author: Akitaka
"""

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from time                    import time
from sklearn.datasets        import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm             import SVC
from sklearn.svm             import OneClassSVM
# from sklearn.metrics         import accuracy_score
from sklearn.metrics         import confusion_matrix
from sklearn.metrics         import classification_report
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
# from sklearn.model_selection import cross_val_score
from my_library              import print_gscv_score
from my_library              import dcv


print(__doc__)
start = time()

# サンプルデータの生成
X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                          n_redundant=0, n_classes=2)
ss = MinMaxScaler()
ss = StandardScaler()
ss.fit(X)
X = ss.transform(X)
X_train, X_test, y_train, y_test = \
 train_test_split(X, y, test_size=0.4)

#%%
range_c = 2**np.arange(  -5, 10, dtype=float)
range_g = 2**np.arange( -10,  0, dtype=float)

param_grid = [{'kernel': ['rbf'], 'C':range_c, 'gamma': range_g}]

score = 'accuracy'
print("# Tuning hyper-parameters for {}".format(score))
print()
print('Search range')
print('c = ', range_c[0], ' ... ',range_c[len(range_c)-1])
print('g = ', range_g[0], ' ... ',range_g[len(range_g)-1])
print()

mod = SVC()
kf = KFold(n_splits=5, shuffle=True)
gscv = GridSearchCV(mod, param_grid, cv=kf, scoring='accuracy')
gscv.fit(X_train, y_train)
print_gscv_score(gscv)

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, gscv.predict(X_test)
print(classification_report(y_true, y_pred))
print(confusion_matrix(y_test, y_pred))
print()

#%%
# Novelty detection by One Class SVM with optimized hyperparameter
clf = OneClassSVM(nu=0.10, kernel=gscv.best_params_['kernel'],
  gamma=gscv.best_params_['gamma'])
clf.fit(X_train)

y_pred = gscv.predict(X_test)    # prediction
reliability = clf.predict(X_test) # outliers = -1
results = np.c_[y_pred, y_test, reliability]
columns=['predicted y','observed y','reliability']
df = pd.DataFrame(results, columns=columns)
print(df)

#%%
for i in range(10):
    dcv(X, y, mod, param_grid)

print('{:.2f} seconds '.format(time() - start))