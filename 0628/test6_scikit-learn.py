# -*- coding: utf-8 -*-
"""
scikit-learn で回帰モデルの結果を評価する
https://pythondatascience.plavox.info/scikit-learn/回帰モデルの評価
本ページでは、Python の機械学習ライブラリの scikit-learn を用いて、回帰モデル (Regression model) の予測精度を評価する方法を紹介します。

回帰モデルの評価にはいくつかの指標があり、本ページでは主要な指標として、MAE, MSE, RMSE, 決定係数の 4 つを紹介します。

Created on Thu Jun 28 15:47:17 2018

@author: Akitaka
"""
# =============================================================================
# 平均絶対誤差 (MAE)
# 平均絶対誤差 (MAE, Mean Absolute Error) は、実際の値と予測値の絶対値を平均したものです。
# MAE が小さいほど誤差が少なく、予測モデルが正確に予測できていることを示し、
# MAE が大きいほど実際の値と予測値に誤差が大きく、予測モデルが正確に予測できていないといえます。
# 平均二乗誤差 (MSE)
# 平均二乗誤差 (MSE, Mean Squared Error) とは、実際の値と予測値の絶対値の 2 乗を平均したものです。
# この為、MAE に比べて大きな誤差が存在するケースで、大きな値を示す特徴があります。
# MAE と同じく、値が大きいほど誤差の多いモデルと言えます。計算式は以下となります。
# 
# 二乗平均平方根誤差 (RMSE)
# MSE の平方根を 二乗平均平方根誤差 (RMSE: Root Mean Squared Error) と呼びます。
# 上記の MSE で、二乗したことの影響を平方根で補正したものです。
# RMSE は、RMSD (Root Mean Square Deviation) と呼ばれることもあります。計算式は以下となります。
# 決定係数 (R2)
# 決定係数 (R2, R-squared, coefficient of determination) は、
# モデルの当てはまりの良さを示す指標で、最も当てはまりの良い場合、1.0 となります
#  (当てはまりの悪い場合、マイナスとなることもあります)。
# 寄与率 (きよりつ) とも呼ばれます。計算式は以下となります。
# =============================================================================
from sklearn.metrics import mean_absolute_error
y_true = [0, 1, 2, 3, 4, 5]
y_pred = [0, 1.2, 2.5, 3.4, 4.6, 5.7]
print(mean_absolute_error(y_true, y_pred))

from sklearn.metrics import mean_squared_error
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(mean_squared_error(y_true, y_pred))

from sklearn.metrics import mean_squared_error
import numpy as np
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(np.sqrt(mean_squared_error(y_true, y_pred)))

from sklearn.metrics import r2_score
y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]
print(r2_score(y_true, y_pred))

