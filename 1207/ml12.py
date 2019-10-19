# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 12:18:31 2018

@author: Akitaka
"""

# 1：ライブラリのインポート--------------------------------
import numpy as np #numpyという行列などを扱うライブラリを利用
import pandas as pd #pandasというデータ分析ライブラリを利用
import matplotlib.pyplot as plt #プロット用のライブラリを利用
from sklearn import cluster, preprocessing, mixture #機械学習用のライブラリを利用
 
# 2：Wineのデータセットを読み込む--------------------------------
df_wine_all=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
#品種(0列、1～3)と色（10列）とプロリンの量(13列)を使用する
df_wine=df_wine_all[[0,10,13]]
df_wine.columns = [u'class', u'color', u'proline']
pd.DataFrame(df_wine)  #この行を実行するとデータが見れる
 
 
# 3：データの整形-------------------------------------------------------
X=df_wine[["color","proline"]]
sc=preprocessing.StandardScaler()
sc.fit(X)
X_norm=sc.transform(X)
 
 
# 4：プロットしてみる------------------------------------------
#%matplotlib inline
 
x=X_norm[:,0]
y=X_norm[:,1]
z=df_wine["class"]
plt.figure(figsize=(10,10))
plt.subplot(3, 1, 1)
plt.scatter(x,y, c=z)
plt.show
 
 
# 5：k-meansを実施---------------------------------
km=cluster.KMeans(n_clusters=3)
z_km=km.fit(X_norm)
 
# 6: 結果をプロット-----------------------------------------------
plt.subplot(3, 1, 2)
plt.scatter(x,y, c=z_km.labels_)
plt.scatter(z_km.cluster_centers_[:,0],z_km.cluster_centers_[:,1],s=250, marker='*',c='red')
plt.show
 
# 解説7：MeanShiftを実施---------------------------------
ms = cluster.MeanShift(seeds=X_norm)
ms.fit(X_norm)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
print(cluster_centers)
 
# 8: 結果をプロット-----------------------------------------------
plt.subplot(3, 1, 3)
plt.scatter(x,y, c=labels)
plt.plot(cluster_centers[0,0], cluster_centers[0,1], marker='*',c='red', markersize=14)
plt.plot(cluster_centers[1,0], cluster_centers[1,1], marker='*',c='red', markersize=14)
plt.show