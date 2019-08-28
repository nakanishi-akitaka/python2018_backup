# -*- coding: utf-8 -*-
"""
1. Classification (k-NN)
2. Applicability Domain (k-NN)
3. Double Cross Validation

Created on Thu Jul 26 15:08:34 2018

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
from my_library              import print_gscv_score
from my_library              import dcv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

#    X = [[0], [1], [2], [3]]
#    y = [0, 0, 1, 1]
#    neigh = KNeighborsClassifier(n_neighbors=3)
#    neigh.fit(X, y) 
#    
#    print(neigh.predict([[1.1]]))
#    
#    print(neigh.predict_proba([[0.9]]))
#    
#    samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
#    neigh = NearestNeighbors(n_neighbors=2)
#    neigh.fit(samples) 
#    
#    print(neigh.kneighbors([[1., 1., 1.]])) 
#    X = [[0., 1., 0.], [1., 0., 1.]]
#    print(neigh.kneighbors(X, return_distance=False))
#    print(neigh.kneighbors(X))
    

#%%

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
# reliability 
y_pred = gscv.predict(X_test)
y_reli = gscv.predict_proba(X_test)[:,1]
y_reli = np.absolute(gscv.predict_proba(X_test)[:,1]-0.5)+0.5

neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(X_train)
dist = np.mean(neigh.kneighbors(X_test)[0], axis=1)
thr = dist.mean() - dist.std()
y_appd = 2 * (dist > thr) -1
#    y_neigh = neigh.kneighbors(X_test)
#    print(y_neigh[0]) # distance 
#    print(y_neigh[1]) # index
#    y_appd = neigh.kneighbors(X_test)[0]
#    print(y_appd)
#    print(type(y_pred),type(y_reli),type(y_appd))
#    print(y_pred.shape,y_reli.shape,y_appd.shape)
#    print(np.mean(y_appd, axis=0)) # mean of column
#    print(np.mean(y_appd, axis=1)) # mean of row 
#    y_appd = np.mean(y_appd, axis=1)
#    y_temp = np.mean(neigh.kneighbors(X_test)[0], axis=1)
#    print(y_temp)
#    print(np.allclose(y_temp, y_appd))
results = np.c_[y_test, y_pred, y_reli, y_appd]
columns=['observed y', 'predicted y', 'reliability', 'applicability']
df = pd.DataFrame(results, columns=columns)
print(df[df.applicability == -1])
# print(df)

#%%
for i in range(10):
    dcv(X, y, mod, param_grid)

print('{:.2f} seconds '.format(time() - start))
