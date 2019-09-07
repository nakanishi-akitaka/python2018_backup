# -*- coding: utf-8 -*-
"""
Example of Machine Learning

1. Classification (k-NN)
2. Applicability Domain (k-NN)
3. Double Cross Validation

Created on Thu Aug  9 10:31:42 2018

@author: Akitaka
"""

import pandas as pd
import numpy as np
from time                    import time
from sklearn.datasets        import make_classification
from sklearn.model_selection import GridSearchCV, ShuffleSplit, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
from sklearn.neighbors       import KNeighborsClassifier
from my_library              import print_gscv_score_clf, dcv_clf, ad_knn

start = time()

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = MinMaxScaler()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier()

range_k = np.arange(  3, 11, dtype=int)

param_grid = [{'n_neighbors':range_k}]

cv = ShuffleSplit(n_splits=5, test_size=0.2)
cv = KFold(n_splits=5, shuffle=True)
gscv = GridSearchCV(model, param_grid, cv=cv)
gscv.fit(X_train, y_train)
print_gscv_score_clf(gscv, X_train, X_test, y_train, y_test, cv)

# Predicted y 
y_pred = gscv.predict(X_test)

# Applicability Domain (inside: +1, outside: -1)
y_appd = ad_knn(X_train, X_test)

results = np.c_[y_pred, y_test, y_appd]
columns=['predicted y','observed y','AD']
df = pd.DataFrame(results, columns=columns)
# print(df[df.AD == 1])
print(df)

if(False):
    dcv_clf(X, y, model, param_grid, 10)

print('{:.2f} seconds '.format(time() - start))
