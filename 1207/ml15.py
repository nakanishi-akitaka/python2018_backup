# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 12:25:32 2018

@author: Akitaka
"""

# 1：ライブラリのインポート--------------------------------
import numpy as np #numpyという行列などを扱うライブラリを利用
import pandas as pd #pandasというデータ分析ライブラリを利用
import matplotlib.pyplot as plt #プロット用のライブラリを利用
from sklearn import cross_validation, preprocessing, linear_model #機械学習用のライブラリを利用
import sklearn
sklearn.__version__
 
#2：Housingのデータセットを読み込む--------------------------------
df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
df.columns=['CRIM','ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
X_rm=df[['RM']].values
X=df.iloc[:, 0:13]
#X=df[['AGE']].values
Y=df['MEDV'].values
 
#3：データの整形-------------------------------------------------------
sc=preprocessing.StandardScaler()
sc.fit(X)
X=sc.transform(X)
 
#4：学習データとテストデータに分割する-------------------------------
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)
 
#5：SGD Regressorを適用する-------------------------------------------
clf = linear_model.SGDRegressor(max_iter=1000)
clf.fit(X_train, Y_train)
 
print("SGDでの係数")
print(clf.intercept_) 
print(clf.coef_) 
 
#解説6：Lasso Regressorを適用する-------------------------------------------
clf_lasso= linear_model.Lasso(alpha=1.0)
clf_lasso.fit(X_train, Y_train)
 
print("\nLassoでの係数")
print(clf_lasso.intercept_) 
print(clf_lasso.coef_) 
 
#解説7：テストデータでの誤差を比較する-------------------------------------------
Y_pred=clf.predict(X_test)
Y_lasso_pred=clf_lasso.predict(X_test)
print("\n「SGDの平均2乗誤差」と「Lassoの平均2乗誤差」")
RMS=np.mean((Y_pred - Y_test) ** 2)
RMS_lasso=np.mean((Y_lasso_pred - Y_test) ** 2)
print(RMS)
print(RMS_lasso)