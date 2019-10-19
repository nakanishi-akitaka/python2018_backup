# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 12:41:01 2018

@author: Akitaka
"""

# 1：ライブラリのインポート--------------------------------
import numpy as np #numpyという行列などを扱うライブラリを利用
import pandas as pd #pandasというデータ分析ライブラリを利用
import matplotlib.pyplot as plt #プロット用のライブラリを利用
from sklearn import cross_validation, preprocessing, decomposition, manifold #機械学習用のライブラリを利用
from sklearn import datasets #使用するデータ
 
# 2：moon型のデータを読み込む--------------------------------
X,Y = datasets.make_moons(n_samples=200, noise=0.05, random_state=0)
 
# 3：データの整形-------------------------------------------------------
sc=preprocessing.StandardScaler()
sc.fit(X)
X_norm=sc.transform(X)
 
# 4：Isomapを実施-------------------------------
isomap = manifold.Isomap(n_neighbors=10, n_components=2)
X_isomap = isomap.fit_transform(X)
 
# 解説5：LLEを実施-------------------------------
lle = manifold.LocallyLinearEmbedding(n_neighbors=10, n_components=2)
X_lle = lle.fit_transform(X)
 
 
# 6: 結果をプロットする-----------------------------
#%matplotlib inline
 
plt.figure(figsize=(10,10))
plt.subplot(3, 1, 1)
plt.scatter(X[:,0],X[:,1], c=Y)
plt.xlabel('x')
plt.ylabel('y')
 
plt.subplot(3, 1, 2)
plt.scatter(X_isomap[:,0],X_isomap[:,1], c=Y)
plt.xlabel('IM-1')
plt.ylabel('IM-2')
 
plt.subplot(3, 1, 3)
plt.scatter(X_lle[:,0],X_lle[:,1], c=Y)
plt.xlabel('LLE-1')
plt.ylabel('LLE-2')
 
plt.show