# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 12:01:18 2018

@author: Akitaka
"""

# 1：ライブラリのインポート--------------------------------
import numpy as np #numpyという行列などを扱うライブラリを利用
import pandas as pd #pandasというデータ分析ライブラリを利用
import matplotlib.pyplot as plt #プロット用のライブラリを利用
from sklearn import cluster, preprocessing #機械学習用のライブラリを利用
from sklearn import datasets #使用するデータ
 
# 2：moon型のデータを読み込む--------------------------------
X,z = datasets.make_moons(n_samples=200, noise=0.05, random_state=0)
 
# 3：データの整形-------------------------------------------------------
sc=preprocessing.StandardScaler()
sc.fit(X)
X_norm=sc.transform(X)
 
 
# 4：プロットしてみる------------------------------------------
#%matplotlib inline
 
x=X_norm[:,0]
y=X_norm[:,1]
plt.figure(figsize=(10,10))
plt.subplot(3, 1, 1)
plt.scatter(x,y, c=z)
plt.show
 
 
# 4：KMeansを実施---------------------------------
km=cluster.KMeans(n_clusters=2)
z_km=km.fit(X_norm)
 
# 5: 結果をプロット-----------------------------------------------
plt.subplot(3, 1, 2)
plt.scatter(x,y, c=z_km.labels_)
plt.scatter(z_km.cluster_centers_[:,0],z_km.cluster_centers_[:,1],s=250, marker='*',c='red')
plt.show
 
 
# 解説6：SpectralClusteringを実施---------------------------------
km=cluster.SpectralClustering(n_clusters=2, affinity="nearest_neighbors")
z_km=km.fit(X_norm)
 
# 7: 結果をプロット-----------------------------------------------
plt.subplot(3, 1, 3)
plt.scatter(x,y, c=z_km.labels_)
plt.show