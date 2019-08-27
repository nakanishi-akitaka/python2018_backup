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
import matplotlib.pyplot as plt
from time                    import time
from sklearn.datasets        import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm             import SVC
from sklearn.svm             import OneClassSVM
from sklearn.metrics         import accuracy_score
from sklearn.metrics         import confusion_matrix
from sklearn.metrics         import classification_report
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
#from sklearn.model_selection import cross_val_score
from my_library              import print_gscv_score


def dcv_clf(mod,param_grid):
    # parameters
    ns_in = 2 # n_splits for inner loop
    ns_ou = 2 # n_splits for outer loop
    
    i = 1 # index of loop
    scores = np.array([]) # list of test scores in outer loop
    kf_ou = KFold(n_splits=ns_ou, shuffle=True)
    
    # [start] outer loop for test of the generalization error
    for train_index, test_index in kf_ou.split(X):
#        start = time()
        X_train, X_test = X[train_index], X[test_index] # inner loop CV
        y_train, y_test = y[train_index], y[test_index] # outer loop 
    
        # [start] inner loop CV for hyper parameter optimization
        kf_in = KFold(n_splits=ns_in, shuffle=True)
        gscv = GridSearchCV(mod, param_grid, cv=kf_in)# , scoring='accuracy'
        gscv.fit(X_train, y_train)
        # [end] inner loop CV for hyper parameter optimization
        
        # test of the generalization error
#        y_pred = gscv.predict(X_test)
#        score = accuracy_score(y_test, y_pred)
        score = gscv.score(X_test, y_test)
        scores = np.append(scores, score)
#        print('dataset: {}/{}  accuracy of inner CV: {:.3f} time: {:.3f} s'.\
#              format(i,ns_ou,score,(time() - start)))
        i+=1
    
    # [end] outer loop for test of the generalization error
    print('  ave, std of accuracy of inner CV: {:.3f} (+/-{:.3f})'\
        .format(scores.mean(), scores.std()*2 ))

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
ldcv = True
if(ldcv):
    for i in range(10):
        dcv_clf(mod, param_grid)

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
results = np.c_[y_pred, y_test, reliability, X_test]
df = pd.DataFrame(results, columns=list('ABCDE'))
df_in_ = df[df.C ==  1]
df_out = df[df.C == -1]
df_bad = df[df.A != df.B]
print('Inlier  sample, number of good/bad predictions: {} {}'.
  format(len(df_in_[df_in_.A == df_in_.B]), len(df_in_[df_in_.A != df_in_.B])))
print('Outlier sample, number of good/bad predictions: {} {}'.
  format(len(df_out[df_out.A == df_out.B]), len(df_out[df_out.A != df_out.B])))
#df = pd.DataFrame(results, columns=['y_pred','y_true','reliability'])
#print(df)

#%%
# visualize 
# ref: https://pythondatascience.plavox.info/matplotlib/散布図
# http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
#
h = .02  # step size in the mesh
x_min, x_max = X_test[:, 0].min() - .2, X_test[:, 0].max() + .2
y_min, y_max = X_test[:, 1].min() - .2, X_test[:, 1].max() + .2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = gscv.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

Z2 = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z2 = Z2.reshape(xx.shape)

plt.figure(figsize=(7,7))
#plt.subplot(1,2,1)
#plt.title("observed value")
#ad = plt.contour(xx, yy, Z, colors=['k', 'k', 'k'],
#            linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
#hp = plt.contour(xx, yy, Z2, levels=[0], linewidths=2, colors='red')
#plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=20, cmap=plt.cm.Paired,
#            edgecolors='k')
#plt.legend([hp.collections[0],ad.collections[0]],
#           ['SVM Hyperplane','OCSVM Applicability Domain'])
#plt.xlim(x_min, x_max) 
#plt.ylim(y_min, y_max) 
#
#plt.subplot(1,2,2)
plt.title("predicted value")
#hp = plt.contour(xx, yy, Z, colors=['k', 'k', 'k'],
#            linestyles=['--', '-', '--'], levels=[-.5, 0, .5])
hp = plt.contour(xx, yy, Z,  levels=[0], colors='k')
ad = plt.contour(xx, yy, Z2, levels=[0], colors='red')
# x = bad prediction
np_bad = np.array(df_bad)
bp = plt.scatter(np_bad[:, 3], np_bad[:, 4], s=100, marker='x',
            zorder=10, edgecolor='k')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, s=20, cmap=plt.cm.Paired,
            edgecolors='k',label='test')

plt.legend([hp.collections[0],ad.collections[0],bp],
           ['SVM Hyperplane','OCSVM Applicability Domain','bad prediction'])
# Circle out the outlier sample
#np_out = np.array(df_out)
#plt.scatter(np_out[:, 3], np_out[:, 4], s=80, facecolors='none',
#            zorder=10, edgecolor='k')
plt.xlim(x_min, x_max) 
plt.ylim(y_min, y_max) 
plt.show()

#%%
print('{:.2f} seconds '.format(time() - start))

