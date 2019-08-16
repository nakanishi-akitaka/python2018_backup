# -*- coding: utf-8 -*-
"""
Pythonで人工知能のWebサービスを実装する方法
http://aiweeklynews.com/archives/48462559.html

Created on Tue Jul 10 11:40:30 2018

@author: Akitaka
"""

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.externals import joblib

#データの読み込み
#df = pd.read_csv('neet.csv')
#neet_data = df.iloc[:,0:5].values
#neet_reslt = df.iloc[:,5].values
from sklearn.datasets import load_iris
iris = load_iris()
neet_data = iris.data
neet_reslt = iris.target


#モデル作成
SVM = svm.SVC(C=3.0, gamma=0.1, probability=True)
SVM.fit(neet_data, neet_reslt)

#モデルを使って予測データを作成
neet_predict = SVM.predict(neet_data)

#cross_validation
print("cross_validation")
print(confusion_matrix(neet_reslt, neet_predict))

#正解率
print("Accuacy")
print(accuracy_score(neet_reslt, neet_predict))

# 予測モデルをシリアライズ
joblib.dump(SVM, 'neet.pkl')