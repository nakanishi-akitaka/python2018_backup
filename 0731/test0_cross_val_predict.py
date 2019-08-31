# -*- coding: utf-8 -*-
"""
CV prediction test

Created on Tue Jul 31 11:43:52 2018

@author: Akitaka
"""

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import KFold

iris = datasets.load_iris()
clf = svm.SVC()

cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
cv = KFold(n_splits=5)

scores = cross_val_score(clf, iris.data, iris.target, cv=cv)
print("cross_val_score")
print(scores)
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

predicted = cross_val_predict(clf, iris.data, iris.target, cv=cv)
print()
print("cross_val_predict")
print("Accuracy: %0.3f" % (metrics.accuracy_score(iris.target, predicted)))

print()
print("self CV with KFold.split")
scores = []
for train_index, test_index in cv.split(iris.data):
    X_train, X_test = iris.data[train_index], iris.data[test_index] # inner loop CV
    y_train, y_test = iris.target[train_index], iris.target[test_index] # outer loop 
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    scores.append(metrics.accuracy_score(y_test,y_pred))
scores = np.array(scores)
print(scores)
print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

#%%

# search range
range_g = 2**np.arange( -20, 11, dtype=float)
print()
print('Search range')
print('g = ', range_g[0], ' ... ',range_g[len(range_g)-1])
print()

param_grid = [{'kernel': ['rbf'], 'gamma': range_g}]

for irs in range(10):
    # cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=irs)
    # -> Error @ cross_val_predict
    cv = KFold(n_splits=5,shuffle=True,random_state=irs)
    gscv = GridSearchCV(clf, param_grid, cv=cv)
    gscv.fit(iris.data, iris.target)
    
    i_best=gscv.cv_results_['params'].index(gscv.best_params_)
    score0=gscv.cv_results_['split0_test_score'][i_best]
    score1=gscv.cv_results_['split1_test_score'][i_best]
    score2=gscv.cv_results_['split2_test_score'][i_best]
    score3=gscv.cv_results_['split3_test_score'][i_best]
    score4=gscv.cv_results_['split4_test_score'][i_best]
    scores = np.array([score0,score1,score2,score3,score4])
    print()
    print("GridSearchCV.cv_results_", scores)
    print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
    
    score = gscv.score(iris.data, iris.target)
    y_pred_gscv = gscv.predict(iris.data)
    y_pred_gscv.sort()
    #print(y_pred_gscv)
    print("GridSeachCV.best_score_ = ", gscv.best_score_)
    print("GridSeachCV.score = ", score)
    y_pred_best = gscv.best_estimator_.predict(iris.data)
    y_pred_best.sort()
    #print("GridSeachCV.predict = ")
    #print(y_pred_best)
    print("y(best) = y(CV)   ?", np.allclose(y_pred_best,y_pred_gscv))
    
    clf = svm.SVC(gamma=gscv.best_params_['gamma'])
    clf.fit(iris.data, iris.target)
    y_pred_opt = clf.predict(iris.data)
    y_pred_opt.sort()
    #print("SVC with opt HP, predict = ")
    #print(y_pred_opt)
    print("y(opt) = y(CV)   ?", np.allclose(y_pred_opt,y_pred_gscv))
    print("y(opt) = y(best) ?", np.allclose(y_pred_opt,y_pred_best))
    
    clf = svm.SVC(gamma=gscv.best_params_['gamma'])
    scores = []
    y_pred_split = []
    for train_index, test_index in cv.split(iris.data):
        X_train, X_test = iris.data[train_index], iris.data[test_index]
        y_train, y_test = iris.target[train_index], iris.target[test_index]
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        y_pred_split.extend(y_pred)
        scores.append(metrics.accuracy_score(y_test,y_pred))
    scores = np.array(scores)
    y_pred_split = np.array(y_pred_split)
    y_pred_split.sort()
    print()
    print("cv.split, scores = ",scores)
    print("Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
    #print("cv.split, predicted y = ")
    #print(y_pred_split)
    print("y(split) = y(CV)   ?", np.allclose(y_pred_split,y_pred_gscv))
    print("y(split) = y(best) ?", np.allclose(y_pred_split,y_pred_best))
    print("y(split) = y(opt)  ?", np.allclose(y_pred_split,y_pred_opt))
    
    print()
    y_pred_cross = cross_val_predict(clf, iris.data, iris.target, cv=cv)
    y_pred_cross.sort()
    #print("cross_val_predict, predicted y = ")
    #print(y_pred_cross)
    print("y(cross) = y(CV)   ?", np.allclose(y_pred_cross,y_pred_gscv))
    print("y(cross) = y(best) ?", np.allclose(y_pred_cross,y_pred_best))
    print("y(cross) = y(opt)  ?", np.allclose(y_pred_cross,y_pred_opt))
    print("y(cross) = y(split)?", np.allclose(y_pred_cross,y_pred_split))
