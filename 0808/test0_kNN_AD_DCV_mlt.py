# -*- coding: utf-8 -*-

"""
Example of Machine Learning - Classification (multi classes)

1. Classification (k-NN)
2. Reliability (k-NN)
3. Applicability Domain (k-NN)
4. Double Cross Validation

Created on Wed Aug  8 10:29:27 2018

@author: Akitaka
"""

import pandas as pd
import numpy as np
from time                    import time
from sklearn.datasets        import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
from sklearn.neighbors       import NearestNeighbors, KNeighborsClassifier
from my_library              import print_gscv_score_clf, dcv_clf

start = time()

X, y = make_classification(n_classes=3,n_informative=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = MinMaxScaler()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%%
mod = KNeighborsClassifier()

range_k = np.arange(  3, 11, dtype=int)

param_grid = [{'n_neighbors':range_k}]

cv = ShuffleSplit(n_splits=5, test_size=0.2)
cv = KFold(n_splits=5, shuffle=True)
gscv = GridSearchCV(mod, param_grid, cv=cv)
gscv.fit(X_train, y_train)
print(gscv.best_params_)
#%%
# Prediction
y_pred = gscv.predict(X_test)
print(y_pred)

# Applicability Domain
neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(X_train)
dist = np.mean(neigh.kneighbors(X_test)[0], axis=1)
thr = dist.mean() - dist.std()
y_appd = 2 * (dist > thr) -1

# Standard Deviation (= Uncertainty <-> Reliability)
y_reli = np.std(y_train[neigh.kneighbors(X_test)[1]], axis=1)

results = np.c_[y_test, y_pred, y_reli, y_appd]
columns=['observed y', 'predicted y', 'Std. Dev. ', 'AD']
df = pd.DataFrame(results, columns=columns)
# print(df)

y_proba = gscv.predict_proba(X_test)[:,1]
y_neigh  = y_train[neigh.kneighbors(X_test)[1]]
y_mean  = np.mean(y_train[neigh.kneighbors(X_test)[1]], axis=1)
y_std   = np.std(y_train[neigh.kneighbors(X_test)[1]], axis=1)
results = np.c_[y_pred, y_proba, y_neigh, y_mean, y_std]
df = pd.DataFrame(results)
print(df)
#%%
if(False):
    dcv_clf(X, y, mod, param_grid, 10)

print('{:.2f} seconds '.format(time() - start))
