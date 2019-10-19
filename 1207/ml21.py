# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 12:40:43 2018

@author: Akitaka
"""

# 1：ライブラリのインポート--------------------------------
import numpy as np #numpyという行列などを扱うライブラリを利用
import pandas as pd #pandasというデータ分析ライブラリを利用
import matplotlib.pyplot as plt #プロット用のライブラリを利用
from sklearn import cross_validation, preprocessing, decomposition #機械学習用のライブラリを利用
 
# 2：Wineのデータセットを読み込む--------------------------------
df_wine_all=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
#品種(0列、1～3)と色（10列）とプロリンの量(13列)を使用する
X=df_wine_all.iloc[:,1:].values
Y=df_wine_all.iloc[:,0].values
 
# 3：データの整形-------------------------------------------------------
sc=preprocessing.StandardScaler()
sc.fit(X)
X=sc.transform(X)
 
# 解説4：主成分分析を実施-------------------------------
pca = decomposition.PCA(n_components=2)
X_transformed = pca.fit_transform(X)
 
# 解説5: 主成分分析の結果-----------------------------
print("主成分の分散説明率")
print(pca.explained_variance_ratio_)
print("固有ベクトル")
print(pca.components_)
 
# 6: 結果をプロットする-----------------------------
#%matplotlib inline
 
plt.figure(figsize=(10,10))
plt.subplot(2, 1, 1)
plt.scatter(X_transformed[:,0],X_transformed[:,1], c=Y)
plt.xlabel('PC1')
plt.ylabel('PC2')
 
plt.subplot(2, 1, 2)
plt.scatter(X[:,9],X[:,12], c=Y)
plt.xlabel('color')
plt.ylabel('proline')
plt.show