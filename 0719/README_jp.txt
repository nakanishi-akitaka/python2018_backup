# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 10:38:48 2018

@author: Akitaka
"""

[1c] Tcへの応用
ref:0718, 0426
example:
0426test1.py

計算パラメータ
探索範囲:
range_c = [i*10**j for j in range(-1,2) for i in range(1,10)]
range_g = [i*10**j for j in range(-1,2) for i in range(1,10)]
※最適化してでてきたγ=90は、推奨値の範囲の外！

CVの分割:
ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

スケーリング
pipe = Pipeline([
('scaler', MinMaxScaler()),
('svr', SVR())])
結果
best_params : {'svr__C': 90, 'svr__gamma': 90, 'svr__kernel': 'rbf'}
learning   score: RMSE, MAE, RMSE/MAE, R^2 = 17.226, 8.559, 2.013, 0.808


[1c2] 1c1をベースに改造していく
example:
test1_0426test1_edit.py

総当たりでTc予測した値(のうち、Tc>100K)をファイルに出力する
出力フォーマットの調整
yyplotと誤差のヒストグラムを出力
探索範囲を推奨値に変更


結果
{'svr__C': 1024.0, 'svr__epsilon': 1.0, 'svr__gamma': 1024.0,
'svr__kernel': 'rbf'}
train data: RMSE, MAE, RMSE/MAE, R^2 = 6.321, 3.881, 1.629, 0.974
大外れはなし
誤差は、大体正規分布




[1c3]
復習
0426test1がメイン
ShuffleSplit, scaling, 説明変数の増加, PCAなどをテスト
ref:
https://mail.google.com/mail/u/0/#sent/RdDgqcJHpWcvcDjPgjkjXHLgLnDfdlQzrnZXHZlrxmfB

Kfold(shuffle=True)とShuffleSplit の違いは07/18を参考
ref:
https://mail.google.com/mail/u/0/#sent/QgrcJHsbjCZNCXqKkMlpLbTXWjKWfzHljSl

test0_{clf,rgr}.pyのGridSearchCVにおいて、分割をcv=5->ShuflleSplitに変更
