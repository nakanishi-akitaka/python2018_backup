# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 11:43:52 2018

@author: Akitaka
"""

[1a2] cross_val_predict 追加計算
example:
test0_cross_val_predict.py

.predictとの比較
(a) GridSearhCV.predict
(b) gscv.best_estimator_.predict
(c) svm.SVC(gamma=gscv.best_params_['gamma']) -> .fit & predict
(d) splitを使った手動CVの予測値
(e) cross_val_predictの予測値

結果
(a) = (b) = (c) =/= (d) = (e) 

考察
GridSearchCV.predict(X)は、最適化したハイパーパラメータを使うが、学習データには何を使っている？
= すべてのX。

以下の３つはすべて同じ結果になる(並べ替えは必要)
(a) y_pred_gscv = gscv.predict(iris.data)

(b) y_pred_best = gscv.best_estimator_.predict(iris.data)

(c) clf = svm.SVC(gamma=gscv.best_params_['gamma'])
    clf.fit(iris.data, iris.target)
    y_pred_opt = clf.predict(iris.data)

https://note.mu/univprof/n/n58399e0a9471
ここでいう、csvファイルは以下の数値で再現できる
CalculatedY.csv = (a),(b),(c)
PredictedYcv.csv = (d),(e)
ただし、
(d) cv=5のように数字の時は無理
(e) cv=ShuffleSplitの時は無理





[1a3] 回帰計算の結果はどう表示するべき？
example:
test0_rgr.py
mylibrary.py

https://note.mu/univprof/n/n7d9eb3ce2c74
■CalculatedY.csv ・・・ それぞれの回帰分析手法におけるモデル構築用データの目的変数の計算値
■PredictedYcv.csv ・・・ それぞれの回帰分析手法におけるモデル構築用データの
　目的変数のクロスバリデーション予測値
■PredictedY1.csv ・・・ それぞれの回帰分析手法における予測用データ1の目的変数の予測値
■StatisticsAll.csv ・・・ それぞれの回帰分析手法におけるモデル構築用データのr^2・RMSE、
　クロスバリデーション後のr^2cv・RMSEcv、予測用データ1のr^2pred・RMSEpredの値

上記4つを計算する関数 print_gscv_score2 を作成。
csvファイルまでは作成せず。
yy-plotは実行。
print_gscv_score2(gscv, X_train, X_test, y_train, y_test, cv)

回帰係数はともかくとして、
DCVの予測値・精度(R^2など)も計算させる？
