# -*- coding: utf-8 -*-
"""
  Double Cross Validation
+ Support Vector Machine
+ One-Class Support Vector Machine 
  Classification
  
Created on Thu Aug  2 15:41:37 2018

@author: Akitaka
"""

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from time                    import time
from sklearn.datasets        import make_classification
from sklearn.model_selection import GridSearchCV, ShuffleSplit, KFold
from sklearn.model_selection import train_test_split
from sklearn.svm             import SVC
from sklearn.svm             import OneClassSVM
# from sklearn.metrics         import accuracy_score
from sklearn.metrics         import confusion_matrix
from sklearn.metrics         import classification_report
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
# from sklearn.model_selection import cross_val_score
from my_library              import print_gscv_score
from my_library              import dcv, optimize_gamma


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
range_c = 2**np.arange(  -5, 11, dtype=float)
range_g = 2**np.arange( -20,  1, dtype=float)

param_grid = [{'kernel': ['rbf'], 'C':range_c, 'gamma': range_g}]

print("# Tuning hyper-parameters")
print()
print('Search range')
print('c = ', range_c[0], ' ... ',range_c[len(range_c)-1])
print('g = ', range_g[0], ' ... ',range_g[len(range_g)-1])
print()

mod = SVC()
n_splits = 5 
cv = ShuffleSplit(n_splits=n_splits, test_size=0.2)
cv = KFold(n_splits=n_splits, shuffle=True)
gscv = GridSearchCV(mod, param_grid, cv=cv)
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
clf = OneClassSVM(nu=0.003, kernel=gscv.best_params_['kernel'],
  gamma=gscv.best_params_['gamma'])
clf.fit(X_train)
reliability1 = clf.predict(X_test) # outliers = -1

# Novelty detection by One Class SVM with optimized hyperparameter
optgamma = optimize_gamma(X_train, range_g) 
clf = OneClassSVM(nu=0.003, kernel=gscv.best_params_['kernel'],
  gamma=optgamma)
clf.fit(X_train)
reliability2 = clf.predict(X_test) # outliers = -1

y_pred = gscv.predict(X_test)    # prediction
results = np.c_[y_pred, y_test, reliability1, reliability2]
columns=['predicted y','observed y','reliability1', 'reliability2']
df = pd.DataFrame(results, columns=columns)
print(df)

#%%
for i in range(1):
    dcv(X, y, mod, param_grid)

print('{:.2f} seconds '.format(time() - start))
