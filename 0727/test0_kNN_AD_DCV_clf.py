# -*- coding: utf-8 -*-
"""
1. Classification (k-NN)
2. Reliability (k-NN)
3. Applicability Domain (k-NN)
4. Double Cross Validation

Created on Fri Jul 27 10:06:59 2018

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
from sklearn.metrics         import confusion_matrix
from sklearn.metrics         import classification_report
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
from my_library              import print_gscv_score, dcv
from sklearn.neighbors       import NearestNeighbors, KNeighborsClassifier

print(__doc__)
start = time()

# サンプルデータの生成
X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                          n_redundant=0, n_classes=2)
ss = MinMaxScaler()
ss = StandardScaler()
ss.fit(X)
X = ss.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#%%
range_k = np.arange(  3, 11, dtype=int)

param_grid = [{'n_neighbors':range_k}]

score = 'accuracy'
print("# Tuning hyper-parameters for {}".format(score))
print()
print('Search range')
print('k = ', range_k[0], ' ... ',range_k[len(range_k)-1])
print()

mod = KNeighborsClassifier()
kf = KFold(n_splits=5, shuffle=True)
gscv = GridSearchCV(mod, param_grid, cv=kf)
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
# Prediction
y_pred = gscv.predict(X_test)

# Applicability Domain
neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(X_train)
dist = np.mean(neigh.kneighbors(X_test)[0], axis=1)
thr = dist.mean() - dist.std()
y_appd = 2 * (dist > thr) -1

# Reliability mean ver. 1
y_reli = np.absolute(gscv.predict_proba(X_test)[:,1]-0.5)+0.5

# Reliability mean ver. 2
#    y_reli = np.absolute(np.mean(y_train[neigh.kneighbors(X_test)[1]], axis=1)
#        -0.5)+0.5

# Reliability std ver. (=/= mean ver.)
#    y_reli = 1- np.std(y_train[neigh.kneighbors(X_test)[1]], axis=1)

results = np.c_[y_test, y_pred, y_reli, y_appd]
columns=['observed y', 'predicted y', 'reliability', 'applicability']
df = pd.DataFrame(results, columns=columns)
print(df)

#%%
for i in range(10):
    dcv(X, y, mod, param_grid)

print('{:.2f} seconds '.format(time() - start))
