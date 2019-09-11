# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 12:54:55 2018

@author: Akitaka
"""

# -*- coding: utf-8 -*-
"""
SVM + OCSVM + contour

Created on Thu Jul 26 09:19:06 2018

@author: Akitaka
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time                    import time
from sklearn.datasets        import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm             import SVC
from sklearn.svm             import OneClassSVM
from sklearn.metrics         import confusion_matrix
from sklearn.metrics         import classification_report
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
#from sklearn.model_selection import cross_val_score
from my_library              import print_gscv_score

print(__doc__)
start = time()

# サンプルデータの生成
X, y = make_classification(n_samples=500, n_features=2, n_informative=2,
                          n_redundant=0, n_classes=2,random_state=40)
ss = MinMaxScaler()
ss = StandardScaler()
ss.fit(X)
X = ss.transform(X)
X_train, X_test, y_train, y_test = \
 train_test_split(X, y, test_size=0.4,random_state=42)

#%%
range_c = 2**np.arange(  -5, 10, dtype=float)
range_g = 2**np.arange( -10,  0, dtype=float)

param_svc = [{'kernel': ['rbf'], 'C':range_c, 'gamma': range_g}]

score = 'accuracy'
print("# Tuning hyper-parameters for {}".format(score))
print()
print('Search range')
print('c = ', range_c[0], ' ... ',range_c[len(range_c)-1])
print('g = ', range_g[0], ' ... ',range_g[len(range_g)-1])
print()

kf = KFold(n_splits=5, shuffle=True)
gscv = GridSearchCV(SVC(), param_svc, cv=kf, scoring='accuracy')
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
from my_library              import optimize_gamma
optgamma = gscv.best_params_['gamma']
range_g = 2**np.arange( -20,  1, dtype=float)
optgamma = optimize_gamma(X_train, range_g) 
clf = OneClassSVM(nu=0.003, kernel=gscv.best_params_['kernel'],
  gamma=optgamma)
clf.fit(X_train)

y_pred = gscv.predict(X_test)    # prediction

from my_library              import ad_knn
# Applicability Domain (inside: +1, outside: -1)
ad_svm = clf.predict(X_test) # outliers = -1
ad_knn = ad_knn(X_train, X_test)

results = np.c_[y_pred, y_test, ad_knn, ad_svm, X_test]

df = pd.DataFrame(results, columns=list('ABCDEF'))
df_knn = df[df.C == -1]
df_svm = df[df.D == -1]
print('AD svm =/= AD knn')
print(df[df.C != df.D])

h = .02  # step size in the mesh
x_min, x_max = X_test[:, 0].min() - .2, X_test[:, 0].max() + .2
y_min, y_max = X_test[:, 1].min() - .2, X_test[:, 1].max() + .2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = gscv.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

Z2 = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z2 = Z2.reshape(xx.shape)

plt.figure(figsize=(6,6))
plt.title("predicted value")
# Circle out the outlier sample
np_knn = np.array(df_knn)
np_svm = np.array(df_svm)
svm = plt.scatter(np_svm[:, 4], np_svm[:, 5], s=200, marker='+', edgecolor='k')
knn = plt.scatter(np_knn[:, 4], np_knn[:, 5], s=200, marker='x', edgecolor='k')
hp = plt.contour(xx, yy, Z,  levels=[0], colors='k')
ad = plt.contour(xx, yy, Z2, levels=[0], colors='red')
all = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, s=20, cmap=plt.cm.Paired,
            edgecolors='k',label='test')
plt.legend([hp.collections[0],ad.collections[0],svm,knn,all],
           ['SVM Hyperplane','OCSVM Applicability Domain',
            'out AD(SVM)','out AD(kNN)','All sample'])
plt.xlim(x_min, x_max) 
plt.ylim(y_min, y_max) 
plt.show()

#%%
print('{:.2f} seconds '.format(time() - start))
