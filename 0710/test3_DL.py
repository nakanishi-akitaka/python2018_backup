# -*- coding: utf-8 -*-
"""
scikit-learnのディープラーニング実装簡単すぎワロタ
http://aiweeklynews.com/archives/50172518.html

Created on Tue Jul 10 12:16:24 2018

@author: Akitaka
"""

#scikit-learnから必要な関数をインポート
import numpy as np 
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import confusion_matrix 
from sklearn.datasets import make_classification

##データの読み込み 
#data = np.loadtxt("neet.csv",delimiter=",", skiprows=1) 
#特徴量データをXに、教師データをyに格納 
#X = data[:, 0:-1]
#y = data[:, -1] 
X, y = make_classification(n_features=5, n_redundant=0, n_informative=2,
                           random_state=0, n_clusters_per_class=1)


#学習データとテストデータに分割 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 

#ニューラルネットワークで学習と評価 
clf = MLPClassifier() 
clf.fit(X_train, y_train)
print(cross_val_score(clf, X_train, y_train))

#混合行列で評価 
y_predict = clf.predict(X_train) 
print(confusion_matrix(y_train, y_predict))


from sklearn import grid_search

#ニューラルネットワークの隠れ層の候補をいろいろ定義
parameters = {'hidden_layer_sizes' : [(100,), (100, 10), (100, 100, 10), 
                                      (100, 100, 100, 10)]}

#ニューラルネットワークのベストな隠れ層を探索
clf = grid_search.GridSearchCV(MLPClassifier(), parameters)
clf.fit(X_train, y_train)
print(clf.best_params_)
