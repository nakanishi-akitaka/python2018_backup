# -*- coding: utf-8 -*-
"""
test of 7 classification method with double cross validation
data: make_classification

Created on Wed Jul 11 16:01:22 2018

@author: Akitaka
"""

print(__doc__)

# modules
# {{{
from time import time
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

#import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# import matplotlib.pyplot as plt
# }}}
#
# functions printing score
#
def print_score(y_test,y_pred): #{{{
    rmse  = np.sqrt(mean_squared_error (y_test,y_pred))
    mae   =         mean_absolute_error(y_test,y_pred)
    r2    =         r2_score           (y_test,y_pred)
    if(mae > 0):
        rmae = np.sqrt(mean_squared_error (y_test,y_pred))/mae
    else:
        rmae = 0.0
#    print('RMSE, MAE, RMSE/MAE, R^2 = {:.3f}, {:.3f}, {:.3f}, {:.3f}'\
#    .format(rmse, mae, rmae, r2))
    print(' {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(rmse, mae, rmae, r2))

#}}}

# Classification
from sklearn.discriminant_analysis    import LinearDiscriminantAnalysis
from sklearn.svm                      import SVC
from sklearn.neighbors                import KNeighborsClassifier
from sklearn.naive_bayes              import GaussianNB
from sklearn.tree                     import DecisionTreeClassifier
from sklearn.ensemble                 import RandomForestClassifier
from sklearn.linear_model             import LogisticRegression
from sklearn                          import metrics

'''
Yingchun Cai et al., J. Chem. Inf. Model.
DOI: 10.1021/acs.jcim.7b00656
ref:06/05,22,28
クラス分類手法は７種類で検討。
 decision tree (DT), k-nearest neighbors (k-NN), logistic regression (LR),
 naive Bayes (NB), neural network (NN), random forest (RF), and
 support vector machine (SVM)

https://note.mu/univprof/n/n38855bb9bfa8
1. 決定木 (Decision Tree, DT)
2. k近傍法によるクラス分類 (k-Nearest Neighbor Classification, kNNC)
3. ロジスティック回帰(LogisticRegression, LR)
4. 単純ベイズ分類器 (Naive Bayes, NB)
5. ニューラルネットワーク (Neural Network, NN)
6. ランダムフォレスト(Random Forests, RF)
7. サポートベクターマシン (Support Vector Machine, SVM)

http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
'''

# サンプルデータの生成
# 1000 samples、5(infomative) + 2(redundant) 
# + 13(independent) =  20 feature のデータを生成
X, y = make_classification(n_samples=1000, n_features=20, n_informative=5,
                          n_redundant=0, n_classes=2, n_clusters_per_class=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# sklearn NO random forest KAIKI
dtc = DecisionTreeClassifier()
knc = KNeighborsClassifier()
lr  = LogisticRegression()
gnb = GaussianNB()
nn  = LinearDiscriminantAnalysis()
rfc = RandomForestClassifier()
svc = SVC()


estimators = {
    'Decision Tree':dtc,
    'k-Nearest Neighbor Classification':knc,
    'LogisticRegression': lr,
    'Gaussian Naive Bayes':gnb,
    'Neural Network': nn,
    'Random Forests':rfc,
    'Support Vector Machine':svc,}

lgraph = False
# KOUSA KENSHO SIMASU

def dcv(mod,param_grid):
    # parameters
    ns_in = 2 # n_splits for inner loop
    ns_ou = 2 # n_splits for outer loop
    
    i = 1 # index of loop
    scores = np.array([]) # list of test scores in outer loop
    kf_ou = KFold(n_splits=ns_ou, shuffle=True)
    
    # [start] outer loop for test of the generalization error
    for train_index, test_index in kf_ou.split(X):
        start = time()
        X_train, X_test = X[train_index], X[test_index] # inner loop CV
        y_train, y_test = y[train_index], y[test_index] # outer loop 
    
        # [start] inner loop CV for hyper parameter optimization
        kf_in = KFold(n_splits=ns_in, shuffle=True)
        gscv = GridSearchCV(mod, param_grid, cv=kf_in, scoring='accuracy')
        gscv.fit(X_train, y_train)
        # [end] inner loop CV for hyper parameter optimization
        
        # test of the generalization error
        y_pred = gscv.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        scores = np.append(scores, score)
        print('dataset: {}/{}  accuracy of inner CV: {:.3f} time: {:.3f} s'.\
              format(i,ns_ou,score,(time() - start)))
        i+=1
    
    # [end] outer loop for test of the generalization error
    print('  ave, std of accuracy of inner CV: {:.3f} (+/-{:.3f})'\
        .format(scores.mean(), scores.std()*2 ))

range_d = np.arange( 1, 10, dtype=int)
param_dt  = [{'max_depth': range_d}]

range_c = 2**np.arange( -2,  5, dtype=float)
param_lr = [{'C': range_c}]

range_n = np.arange( 1,  10, dtype=int)
param_knc = [{'n_neighbors': range_n}]

range_c = 2**np.arange( -2,  5, dtype=float)
param_svm = [{'C': range_c}]

test = {
    'DT ':{'name':'Decision Tree         ','model':dtc,'param':param_dt },
    'LR ':{'name':'LogisticRegression    ','model':lr ,'param':param_lr },
    'kNN':{'name':'k-Nearest Neighbor    ','model':knc,'param':param_knc},
    'RF ':{'name':'Random Forest         ','model':dtc,'param':param_dt },
    'SVM':{'name':'Support Vector Machine','model':svc,'param':param_svm},
    }

for key, value in test.items():
    print(test[key]['name'])
    dcv(test[key]['model'],test[key]['param'])
print()
#        'Decision Tree':dtc,
#        'k-Nearest Neighbor Classification':knc,
#        'LogisticRegression': lr,
#        'Gaussian Naive Bayes':gnb,
#        'Neural Network': nn,
#        'Random Forests':rfc,
#        'Support Vector Machine':svc,
print('Accuracy, Precision, Recall, F1-score, False Positive Rate, \
True Positive Rate')
print('A,     P,     R,     F1,    FPR,   TPR')
for k,v in estimators.items():
    start = time()

    # step 1. model
    mod = v

    # step 2. learning -> score
    mod.fit(X_train, y_train)

    y_pred = mod.predict(X_test)
    # step 3. score
#    print('        confusion_matrix')
#    print( metrics.confusion_matrix(y, y_pred))
#    print('{:.3f}'.format(metrics.accuracy_score(y, y_pred)))
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
    print('{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:}'.format(
            metrics.accuracy_score(y_test, y_pred),
            metrics.precision_score(y_test, y_pred),
            metrics.recall_score(y_test, y_pred),
            metrics.f1_score(y_test, y_pred),
            fp/(tn+fp), tp/(fn+tp), k))
#    print('True Positive, False Positive, False Negative, True Positive')
#    print(tn, fp, fn, tp)
#    print('False Positive Rate, True Positive Rate')
#    print(fp/(tn+fp), tp/(fn+tp))