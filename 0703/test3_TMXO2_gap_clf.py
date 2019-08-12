# -*- coding: utf-8 -*-
"""
Machine learning energy gap of ABO2 (classification)

Created on Tue Jul  3 13:41:58 2018

@author: Akitaka
"""


print(__doc__)

# modules
# {{{
from time import time
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
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

# read data from csv file
name = '../0625/test1.csv'
data = np.array(pd.read_csv(name))[:,:]
y=data[:,8]
X=data[:,0:2]
y = 1.0 * (y > 0)

# Classification
from sklearn.discriminant_analysis    import LinearDiscriminantAnalysis
from sklearn.svm                      import LinearSVC
from sklearn.svm                      import SVC
from sklearn.discriminant_analysis    import QuadraticDiscriminantAnalysis
from sklearn.neighbors                import KNeighborsClassifier
from sklearn.naive_bayes              import GaussianNB
from sklearn.tree                     import DecisionTreeClassifier
from sklearn.ensemble                 import RandomForestClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process         import GaussianProcessClassifier
from sklearn.ensemble                 import BaggingClassifier
from sklearn.ensemble                 import AdaBoostClassifier
'''
https://note.mu/univprof/n/n38855bb9bfa8
1. 線形判別分析(Linear Discriminant Analysis, LDA)
2. 線形サポートベクターマシン (Linear Support Vector Machine, LSVM)
3. 非線形サポートベクターマシン (Non-Linear Support Vector Machine, NLSVM)
4. 二次判別分析 (Quadratic Discriminant Analysis, QDA)
5. k近傍法によるクラス分類 (k-Nearest Neighbor Classification, kNNC)
6. 単純ベイズ分類器 (Naive Bayes, NB)
7. 決定木 (Decision Tree, DT)
8. ランダムフォレスト(Random Forests, RF)
9. Gaussian Process Classification (GPC)
10. LDAに基づくバギング(アンサンブル) (Bagging[LDA])
11. LSVMに基づくバギング(アンサンブル) (Bagging[LSVM])
12. NLSVMに基づくバギング(アンサンブル) (Bagging[NLSVM])
13. QDAに基づくバギング(アンサンブル) (Bagging[QDA])
14. kNNCに基づくバギング(アンサンブル) (Bagging[kNNC])
15. NBに基づくバギング(アンサンブル) (Bagging[NB])
16. DTに基づくバギング(アンサンブル) (Bagging[DT])
17. GPCに基づくバギング(アンサンブル) (Bagging[GPC])
18. LSVMに基づくAdaptive Boosting (AdaBoost[LSVM])
19. NLSVMに基づくAdaptive Boosting (AdaBoost[NLSVM])
20. NBに基づくAdaptive Boosting (AdaBoost[NB])
21. DTに基づくAdaptive Boosting (AdaBoost[DT])

http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html
http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html
'''

# sklearn NO random forest KAIKI
lda = LinearDiscriminantAnalysis()
svl = LinearSVC()
svc = SVC()
qda = QuadraticDiscriminantAnalysis()
knc = KNeighborsClassifier()
gnb = GaussianNB()
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
gpc = GaussianProcessClassifier(1.0 * RBF(1.0))
bct = BaggingClassifier(lda)
bcl = BaggingClassifier(svl)
bcs = BaggingClassifier(svc)
bcq = BaggingClassifier(qda)
bck = BaggingClassifier(knc)
bcb = BaggingClassifier(gnb)
bcd = BaggingClassifier(dtc)
bcg = BaggingClassifier(gpc)
abl = AdaBoostClassifier(svl,algorithm="SAMME")
abs = AdaBoostClassifier(svc,algorithm="SAMME")
abb = AdaBoostClassifier(gnb,algorithm="SAMME")
abd = AdaBoostClassifier(dtc,algorithm="SAMME")

# ValueError: BaseClassifier in AdaBoostClassifier ensemble
# is worse than random, ensemble can not be fit.
abl = svl # dummy

estimators = {
        'Linear Discriminant Analysis':lda,
        'Linear Support Vector Machine':svl,
        'Non-Linear Support Vector Machine':svc,
        'Quadratic Discriminant Analysis':qda,
        'k-Nearest Neighbor Classification':knc,
        'Gaussian Naive Bayes':gnb,
        'Decision Tree':dtc,
        'Random Forests':rfc,
        'Gaussian Process Classification':gpc,
        'Bagging[LDA]':bct,
        'Bagging[LSVM]':bcl,
        'Bagging[NLSVM]':bcs,
        'Bagging[QDA]':bcq,
        'Bagging[kNNC]':bck,
        'Bagging[NB]':bcb,
        'Bagging[DT]':bcd,
        'Bagging[GPC]':bcg, 
        'AdaBoost[LSVM]':abl,
        'AdaBoost[NLSVM]':abs,
        'AdaBoost[NB]':abb,
        'AdaBoost[DT]':abd,}

lgraph = False
# KOUSA KENSHO SIMASU

print('Accuracy, Precision, Recall, F1-score, AUC')

names = []
accs = []
pres = []
recs = []
f1ss = []
aucs = []
for k, v in estimators.items():
    start = time()

    # step 1. model
    mod = v

    # step 2. learning -> score
    mod.fit(X, y)
    y_pred = mod.predict(X)
    kfold = KFold(n_splits=3, random_state=42)
    accuracy  = cross_val_score(mod,X, y, cv = kfold, scoring = "accuracy")
    precision = cross_val_score(mod,X, y, cv = kfold, scoring = "precision")
    recall    = cross_val_score(mod,X, y, cv = kfold, scoring = "recall")
    f1        = cross_val_score(mod,X, y, cv = kfold, scoring = "f1")
    roc_auc   = cross_val_score(mod,X, y, cv = kfold, scoring = "roc_auc")
    names.append(k)
    accs.append(accuracy)
    pres.append(precision)
    recs.append(recall)
    f1ss.append(f1)
    aucs.append(roc_auc)

for i in range(len(names)):
    '''
    print('{:.2f} (+/- {:.2f}), {:.2f} (+/- {:.2f}), {:.2f} (+/- {:.2f}),\
 {:.2f} (+/- {:.2f}), {:.2f} (+/- {:.2f}), {:}'.format(
            accs[i].mean(), accs[i].std(),
            pres[i].mean(), pres[i].std(),
            recs[i].mean(), recs[i].std(),
            f1ss[i].mean(), f1ss[i].std(),
            aucs[i].mean(), aucs[i].std(),
            names[i]))
    '''
    print('{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:}'.format(
            accs[i].mean(),
            pres[i].mean(),
            recs[i].mean(),
            f1ss[i].mean(),
            aucs[i].mean(),
            names[i]))

