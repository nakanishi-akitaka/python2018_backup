# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 10:33:05 2018

@author: Akitaka
"""

[1c1] 水素化物のTc予測プログラムアップデート
example:
test1_to_csv.py
example:
test2_Tc_SVM.py

更新の余地リストから、以下の内容をアップデート
* ShuffleSplit, KFold(shuffle=True),split数の変化
* scaling = MinMax or Standard

以下の通り、両方のせるだけの簡単改造
使いたい方を後に移動させる
    n_splits = 5 
    cv = ShuffleSplit(n_splits=n_splits, test_size=0.2)
    cv = KFold(n_splits=n_splits, shuffle=True)




[1d1] SVM+OCSVM+ダブルクロスバリデーション/二重交差検証/DCV プログラムをアップデート
ref:
0713, 0716, 0717,0718
http://univprof.com/archives/16-06-12-3889388.html
https://datachemeng.com/doublecrossvalidation/

DCVプログラムの方向性について　総当たりの奴を参考に
https://note.mu/univprof/n/n58399e0a9471
https://note.mu/univprof/n/n38855bb9bfa8


example:
test3_SVM_OCSVM_DCV_clf.py
test3_SVM_OCSVM_DCV_rgr.py

* 分類だけでなく、回帰も作成。
DCVの部分は、どちらでも使えるように作れる！？
    gscv = GridSearchCV(mod, param_grid, cv=kf_in, scoring='accuracy')
    ...
    y_pred = gscv.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    →
    gscv = GridSearchCV(mod, param_grid, cv=kf_in)
    ...
    score = gscv.score(X_test, y_test)
どちらの場合も、modに入れた機械学習方法の.scoreメソッドを用いることになる！
他の評価方法が欲しければ別だが。
→　それはそれで改造すればいい。


* 予測値、観測値、信頼性をまとめてデータフレームに変換
ref:
https://note.mu/univprof/n/ndb7f96e13f59
ここでは、予測値と信頼性を別々のcsvファイルに保存
また、観測値はない。純粋な予測。


？？？　DCVの結果はどう表示するべき？？？
https://note.mu/univprof/n/n58399e0a9471
■CalculatedY.csv ・・・ それぞれの回帰分析手法におけるモデル構築用データの目的変数の計算値
■PredictedYcv.csv ・・・ それぞれの回帰分析手法におけるモデル構築用データの目的変数のクロスバリデーション予測値
■PredictedY1.csv ・・・ それぞれの回帰分析手法における予測用データ1の目的変数の予測値
■PredictedY2.csv ・・・ それぞれの回帰分析手法における予測用データ2の目的変数の予測値
■StatisticsAll.csv ・・・ それぞれの回帰分析手法におけるモデル構築用データのr^2・RMSE、クロスバリデーション後のr^2cv・RMSEcv、予測用データ1のr^2pred・RMSEpredの値
■StandardRegressionCoefficients.csv ・・・ 線形回帰分析手法(OLS, PLS, RR, LASSO, NE, LSVR)における標準回帰係数
■PredictedYdcv.csv ・・・ それぞれの回帰分析手法におけるモデル構築用データの目的変数のダブルクロスバリデーション予測値 http://univprof.com/archives/16-06-12-3889388.html (省略も可能)
■StatisticsDcvAll.csv ・・・ それぞれの回帰分析手法におけるモデル構築用データのダブルクロスバリデーション後のr^2dcv・RMSEdcvの値 (省略も可能)
