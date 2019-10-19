# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 12:40:48 2018

@author: Akitaka
"""

# 1：ライブラリのインポート--------------------------------
import numpy as np #numpyという行列などを扱うライブラリを利用
import pandas as pd #pandasというデータ分析ライブラリを利用
import matplotlib.pyplot as plt #プロット用のライブラリを利用
from sklearn import cross_validation, preprocessing, decomposition #機械学習用のライブラリを利用
from sklearn import datasets #使用するデータ
 
# 2：moon型のデータを読み込む--------------------------------
X,Y = datasets.make_moons(n_samples=200, noise=0.05, random_state=0)
 
# 3：データの整形-------------------------------------------------------
sc=preprocessing.StandardScaler()
sc.fit(X)
X_norm=sc.transform(X)
 
# 4：主成分分析を実施-------------------------------
pca = decomposition.PCA(n_components=2)
X_pca = pca.fit_transform(X)
 
# 解説5：カーネル主成分分析を実施-------------------------------
kpca = decomposition.KernelPCA(n_components=2,  kernel='rbf', gamma=20.0)
X_kpca = kpca.fit_transform(X)
 
 
# 6: 結果をプロットする-----------------------------
#%matplotlib inline
 
plt.figure(figsize=(10,10))
plt.subplot(3, 1, 1)
plt.scatter(X[:,0],X[:,1], c=Y)
plt.xlabel('x')
plt.ylabel('y')
 
plt.subplot(3, 1, 2)
plt.scatter(X_pca[:,0],X_pca[:,1], c=Y)
plt.xlabel('PC1')
plt.ylabel('PC2')
 
plt.subplot(3, 1, 3)
plt.scatter(X_kpca[:,0],X_kpca[:,1], c=Y)
plt.xlabel('K-PC1')
plt.ylabel('K-PC2')
 
plt.show