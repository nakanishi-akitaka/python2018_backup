# -*- coding: utf-8 -*-
"""
test of scaler

Created on Fri Aug  3 13:09:02 2018

@author: Akitaka
"""

import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.datasets        import make_regression
from sklearn.model_selection import train_test_split
from sklearn.svm             import SVR
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import make_pipeline

X, y = make_regression(n_samples=10, n_features=2, n_informative=2,random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train1 = scaler.fit_transform(X_train)
X_test1 = scaler.transform(X_test)

scaler2 = StandardScaler()
scaler2.fit(X_train)
X_train2 = scaler.transform(X_train)
X_test2 = scaler2.transform(X_test)

print(np.allclose(X_test1,X_test2))
print(np.allclose(X_train1,X_train2))

model = SVR()
model.fit(X_train1,y_train)
y_pred1 = model.predict(X_test1)

pipe = make_pipeline(StandardScaler(),SVR())
pipe.fit(X_train,y_train)
y_pred2 = pipe.predict(X_test)
print(np.allclose(y_pred1,y_pred2))
