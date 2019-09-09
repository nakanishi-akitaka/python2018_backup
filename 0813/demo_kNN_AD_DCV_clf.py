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
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics         import confusion_matrix, accuracy_score
from sklearn.neighbors       import NearestNeighbors




def print_gscv_score_clf(gscv, X_train, X_test, y_train, y_test, cv):
    print()
    print("Best parameters set found on development set:")
    print(gscv.best_params_)
    y_calc = gscv.predict(X_train)
    tn, fp, fn, tp = confusion_matrix(y_train, y_calc).ravel()
    print('C:  TP, FP, FN, TN, Acc. = {0}, {1}, {2}, {3}, {4:.3f}'.\
          format(tp, fp, fn, tn, accuracy_score(y_train, y_calc)))

    y_incv = cross_val_predict(gscv, X_train, y_train, cv=cv)
    tn, fp, fn, tp = confusion_matrix(y_train, y_incv).ravel()
    print('CV: TP, FP, FN, TN, Acc. = {0}, {1}, {2}, {3}, {4:.3f}'.\
          format(tp, fp, fn, tn, accuracy_score(y_train, y_incv)))

    y_pred = gscv.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print('P:  TP, FP, FN, TN, Acc. = {0}, {1}, {2}, {3}, {4:.3f}'.\
          format(tp, fp, fn, tn, accuracy_score(y_test, y_pred)))
    print()



def dcv_clf(X,y,mod,param_grid,niter):
    # parameters
    ns_in = 3 # n_splits for inner loop
    ns_ou = 3 # n_splits for outer loop
    scores = np.zeros((niter,5))
    for iiter in range(niter):
        ypreds = np.array([]) # list of predicted y in outer loop
        ytests = np.array([]) # list of y_test in outer loop
        kf_ou = KFold(n_splits=ns_ou, shuffle=True)
    
        # [start] outer loop for test of the generalization error
        for train_index, test_index in kf_ou.split(X):
            X_train, X_test = X[train_index], X[test_index] # inner loop CV
            y_train, y_test = y[train_index], y[test_index] # outer loop 
        
            # [start] inner loop CV for hyper parameter optimization
            kf_in = KFold(n_splits=ns_in, shuffle=True)
            gscv = GridSearchCV(mod, param_grid, cv=kf_in)
            gscv.fit(X_train, y_train)
            # [end] inner loop CV for hyper parameter optimization
            
            # test of the generalization error
            ypred = gscv.predict(X_test)
            ypreds = np.append(ypreds, ypred)
            ytests = np.append(ytests, y_test)
        
        # [end] outer loop for test of the generalization error
        tn, fp, fn, tp = confusion_matrix(ytests, ypreds).ravel()
        acc = accuracy_score(ytests, ypreds)
        scores[iiter,:] = np.array([tp,fp,fn,tn,acc])

    means, stds = np.mean(scores, axis=0),np.std(scores, axis=0)
    print()
    print('Double Cross Validation')
    print('In {:} iterations, average +/- standard deviation'.format(niter))
    print('TP   DCV: {:.3f} (+/-{:.3f})'.format(means[0], stds[0]))
    print('FP   DCV: {:.3f} (+/-{:.3f})'.format(means[1], stds[1]))
    print('FN   DCV: {:.3f} (+/-{:.3f})'.format(means[2], stds[2]))
    print('TN   DCV: {:.3f} (+/-{:.3f})'.format(means[3], stds[3]))
    print('Acc. DCV: {:.3f} (+/-{:.3f})'.format(means[4], stds[4]))




# Applicability Domain with k-Nearest Neighbor
def ad_knn(X_train, X_test):
    n_neighbors = 5      # number of neighbors
    r_ad = 0.9           # ratio of X_train inside AD / all X_train

    neigh = NearestNeighbors(n_neighbors=n_neighbors+1)
    neigh.fit(X_train)
    dist_list = np.mean(neigh.kneighbors(X_train)[0][:,1:], axis=1)
    dist_list.sort()
    ad_thr = dist_list[round(X_train.shape[0] * r_ad) - 1]
    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(X_train)
    dist = np.mean(neigh.kneighbors(X_test)[0], axis=1)
    y_appd = 2 * (dist < ad_thr) -1

    return y_appd

start = time()

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier()

range_k = np.arange(  3, 11, dtype=int)

param_grid = [{'n_neighbors':range_k}]

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
# print(df)

dcv_clf(X, y, model, param_grid, 10)

print('{:.2f} seconds '.format(time() - start))
