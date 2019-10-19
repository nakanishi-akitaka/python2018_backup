# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 12:32:12 2018

@author: Akitaka
"""

# 1：ライブラリのインポート--------------------------------
import numpy as np #numpyという行列などを扱うライブラリを利用
import pandas as pd #pandasというデータ分析ライブラリを利用
import matplotlib.pyplot as plt #プロット用のライブラリを利用
from sklearn import cross_validation, preprocessing, linear_model, svm #機械学習用のライブラリを利用
 
# 2：Housingのデータセットを読み込む--------------------------------
df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header=None, sep='\s+')
df.columns=['CRIM','ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
X_rm=df[['RM']].values
X=df.iloc[:, 0:13]
#X=df[['AGE']].values
Y=df['MEDV'].values
 
# 3：データの整形-------------------------------------------------------
sc=preprocessing.StandardScaler()
sc.fit(X)
X=sc.transform(X)
sc.fit(X_rm)
X_rm=sc.transform(X_rm)
 
# 4：学習データとテストデータに分割する-------------------------------
X_rm_train, X_rm_test, Y_train, Y_test = cross_validation.train_test_split(X_rm, Y, test_size=0.5, random_state=0)
 
# 5：SGD Regressorを適用する-------------------------------------------
clf_rm = linear_model.SGDRegressor(max_iter=1000)
clf_rm.fit(X_rm_train, Y_train)
 
# 解説6：SVR linear Regressorを適用する-------------------------------------------
clf_svr = svm.SVR(kernel='linear', C=1e3, epsilon=2.0)
clf_svr.fit(X_rm_train, Y_train)
 
# 7：結果をプロットする------------------------------------------------
#%matplotlib inline
 
line_X=np.arange(-4, 4, 0.1) #3から10まで1刻み
line_Y_sgd=clf_rm.predict(line_X[:, np.newaxis])
line_Y_svr=clf_svr.predict(line_X[:, np.newaxis])
plt.figure(figsize=(10,10))
plt.subplot(2, 1, 1)
plt.scatter(X_rm_train, Y_train, c='b', marker='s')
plt.plot(line_X, line_Y_sgd, c='r')
plt.plot(line_X, line_Y_svr, c='g')
plt.show
 
# 8：誤差-------------------------------------------------
Y_pred_sgd=clf_rm.predict(X_rm_test)
Y_pred_svr=clf_svr.predict(X_rm_test)
print("\n「SGDの平均2乗誤差」と「SVRの平均二乗誤差」")
RMS_sgd=np.mean((Y_pred_sgd - Y_test) ** 2)
RMS_svr=np.mean((Y_pred_svr - Y_test) ** 2)
print(RMS_sgd)
print(RMS_svr)