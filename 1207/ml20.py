# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 12:36:14 2018

@author: Akitaka
"""

# 1：ライブラリのインポート--------------------------------
import numpy as np #numpyという行列などを扱うライブラリを利用
import pandas as pd #pandasというデータ分析ライブラリを利用
import matplotlib.pyplot as plt #プロット用のライブラリを利用
from sklearn import cross_validation, preprocessing, ensemble, tree #機械学習用のライブラリを利用
 
# 2：データの生成--------------------------------
# Generate sample data
numSamples=80
x = np.sort(5 * np.random.rand(numSamples, 1), axis=0)
y = np.sin(x).ravel()
y= y + 0.2 * (np.random.randn(numSamples))
y=y.ravel()
 
x_true=np.arange(0, 5.0, 0.1) # 3から10まで1刻み
y_true = np.sin(x_true).ravel()
 
 
# 解説3：BaggingRegressorを適用する--------------------------------
reg = ensemble.BaggingRegressor(tree.DecisionTreeRegressor(), n_estimators=100, max_samples=0.3)
x_true=x_true.reshape((-1, 1))
y_reg = reg.fit(x, y).predict(x_true.reshape(-1, 1))
 
# 4：プロットしてみる------------------------------------------
#%matplotlib inline
 
plt.scatter(x, y, color='darkorange', label='data')
plt.plot(x_true, y_true, color='navy', label='sin')
plt.plot(x_true, y_reg, color='c', label='BaggingRegressor')
plt.legend()
plt.show()