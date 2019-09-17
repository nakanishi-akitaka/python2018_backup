# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:07:04 2018

@author: Akitaka
"""

サンプル解読
Downloads/python/1009/y_randomization-master/demo_y_randamization.ipynb
1.PLS+CVをやる
2.yをnp.random.permutationで入れ替える
3.通常のPLS+CVとほぼ同じ手順を実行　※X_testに対する.predictがなくなっている
4.{計算,CV}に対するyの予測値に対する{R2,RMSE,MAE}を計算
5.2-4を複数回繰り返す
6.4.の分布と平均値を図示する。
7.通常の計算1.での{計算,CV,test}に対するyの予測値に対する{R2,RMSE,MAE}を表示。

ref:
permutationについて検索
numpyのshuffleとpermutationの違い
http://kaisk.hatenadiary.com/entry/2014/10/30/170522
公式ドキュメント　numpy.random.permutation
https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.permutation.html


[1a3] kNNでテスト
Downloads/python/1011/kNN_y_randamization.ipynb

水素化物ではなく、データセット作成
ランダム後の予測値は、CVはなしでCのみ(通常のy=model.predict(X))
→
# modeling and prediction
Best parameters set found on development set:
{'n_neighbors': 6}
C:  RMSE, MAE, R^2 = 17.470, 12.516, 0.971
CV: RMSE, MAE, R^2 = 22.122, 16.029, 0.953
P:  RMSE, MAE, R^2 = 19.912, 15.064, 0.958

# y-randomization
In 10 iterations, average +/- standard deviation
RMSE: 96.776 (+/-0.602)
MAE: 77.404 (+/-0.655)
R^2: 0.097 (+/-0.011)
