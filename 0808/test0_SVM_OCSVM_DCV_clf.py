# -*- coding: utf-8 -*-
"""
Example of Machine Learning - Classification

1. Classification (Support Vector Machine)
2. Applicability Domain (One-Class Support Vector Machine )
3. Double Cross Validation

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
from sklearn.svm             import SVC, OneClassSVM
from my_library              import print_gscv_score_clf, dcv_clf, optimize_gamma

print(__doc__)

start = time()

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = MinMaxScaler()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
 
#%%
mod = SVC()

range_c = 2**np.arange(  -5+15, 11, dtype=float)
range_g = 2**np.arange( -20,  1, dtype=float)

param_grid = [{'kernel': ['rbf'], 'C':range_c, 'gamma': range_g}]

cv = ShuffleSplit(n_splits=5, test_size=0.2)
cv = KFold(n_splits=5, shuffle=True)
gscv = GridSearchCV(mod, param_grid, cv=cv)
gscv.fit(X_train, y_train)
print_gscv_score_clf(gscv, X_train, X_test, y_train, y_test, cv)

#%%
# Applicability Domains with One-Class Support Vector Machine
optgamma = gscv.best_params_['gamma']
optgamma = optimize_gamma(X_train, range_g) 
clf = OneClassSVM(nu=0.003, kernel=gscv.best_params_['kernel'], gamma=optgamma)
clf.fit(X_train)
ad = clf.predict(X_test) # outliers = -1

y_pred = gscv.predict(X_test)    # prediction
results = np.c_[y_pred, y_test, ad]
columns=['predicted y','observed y','AD']
df = pd.DataFrame(results, columns=columns)
# print(df)

#%%
if(False):
    dcv_clf(X, y, mod, param_grid, 10)

print('{:.2f} seconds '.format(time() - start))
