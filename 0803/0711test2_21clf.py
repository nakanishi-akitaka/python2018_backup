# -*- coding: utf-8 -*-
"""
test of 21 classification method
data: make_classification

Created on Wed Jul 11 14:41:42 2018

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
#from sklearn.model_selection import KFold
#from sklearn.model_selection import cross_val_score
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
from sklearn                          import metrics

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

# サンプルデータの生成
# 1000 samples、5(infomative) + 2(redundant) 
# + 13(independent) =  20 feature のデータを生成
X, y = make_classification(n_samples=1000, n_features=20, n_informative=5,
                          n_redundant=0, n_classes=2, n_clusters_per_class=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

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