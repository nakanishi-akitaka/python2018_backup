# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 15:13:28 2018

@author: Akitaka
"""

import pandas as pd
import numpy as np
from time                    import time
from sklearn.datasets        import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm             import SVC
from sklearn.svm             import OneClassSVM
from sklearn.metrics         import confusion_matrix
from sklearn.metrics         import classification_report
from sklearn.preprocessing   import StandardScaler
#from sklearn.model_selection import cross_val_score

def print_gscv_score(gscv): #{{{
    print("Best parameters set found on development set:")
    print()
    print(gscv.best_params_)
    print()
    print("Grid scores on development set:")
    print()
#    means = gscv.cv_results_['mean_test_score']
#    stds = gscv.cv_results_['std_test_score']
#    for mean, std, params in zip(means, stds, gscv.cv_results_['params']):
#        print("{:.3f} (+/-{:.03f}) for {:}".format(mean, std * 2, params))

print(__doc__)
start = time()

# サンプルデータの生成
X, y = make_classification(n_samples=500, n_features=5, n_informative=5,
                          n_redundant=0, n_classes=2, n_clusters_per_class=5)
ss = StandardScaler()
ss.fit(X)
X = ss.transform(X)
X_train, X_test, y_train, y_test = \
 train_test_split(X, y, test_size=0.4,random_state=42)

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
clf = OneClassSVM(nu=0.10, kernel=gscv.best_params_['kernel'],
  gamma=gscv.best_params_['gamma'])
clf.fit(X_train)

y_pred = gscv.predict(X_test)    # prediction
reliability = clf.predict(X_test) # outliers = -1
results = np.c_[y_pred, y_test, reliability]
#print('y_predicted, y_true, outliers = -1')
#print(y_tot)
#print()

#%%
df = pd.DataFrame(results, columns=list('ABC'))
df_in_ = df[df.C ==  1]
df_out = df[df.C == -1]
print('Inlier  sample, number of good/bad predictions: {} {}'.
  format(len(df_in_[df_in_.A != df_in_.B]), len(df_in_[df_in_.A == df_in_.B])))
print('Outlier sample, number of good/bad predictions: {} {}'.
  format(len(df_out[df_out.A != df_out.B]), len(df_out[df_out.A == df_out.B])))

#df = pd.DataFrame(results, columns=['y_pred','y_true','reliability'])
#print(df)
#%%
print('{:.2f} seconds '.format(time() - start))
