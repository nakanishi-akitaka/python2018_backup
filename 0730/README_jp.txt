# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 10:56:15 2018

@author: Akitaka
"""


Xの変数を適切に非線形変換して線形の回帰モデルを作るとモデルの適用範囲・適用領域が広がるかも！
[検証結果とPythonプログラムあり]
https://datachemeng.com/variabletransformationad/
X と y との間の非線形性に対応する２つの方法
    非線形の回帰分析手法を用いる
        非線形回帰モデルによってモデル化された X と y との間の非線形性と、
        実際の非線形性とが一致するとは限りません
        ->  モデルが外挿をできにくくなる (モデルの適用範囲・適用領域が狭くなる)
    X を非線形関数で適切に変換する
        モデルが外挿しやすそう -> モデルの適用範囲が広くなる
データ解析により、少なくとも上の数値シミュレーションデータにおいては、
X を非線形関数で適切に変換して線形の手法で回帰モデルを作ったとき、
非線形の回帰分析手法で回帰モデルを作ったときと比べてモデルの適用範囲が広くなる、
という仮説を、検証できました。
課題
    1.モデルの適用範囲が広くなるためには、(非線形変換後の) X と y との本来の線形関係を
      モデルで表現しなければならないが、特に X の変数が多いときや X 間の相関が高いときなど、
      正しく線形モデルを構築できるのか？
    2.そもそも、どうやって “X を非線形関数で適切に変換” するのか？
1. の課題についてはダブルクロスバリデーションによる変数選択およびモデル選択、
2. の課題については いわゆる理論モデル・物理モデル・第一原理モデル、
といった解決の方向性があると考えています。
理論と統計の融合により、モデルの適用範囲が広がることで貢献できる可能性があることが示唆された、
といった感じです。

example:
test0_variable_transform_ad.py
4つも図を描く
1.trainデータの予測値
2.trainデータのCVでの予測値
3.testデータの予測値 out AD
4.testデータの予測値 in AD



[1a3] cross_val_predict 1.スコアの再現
GridSearchCVの予測値
2.は何なのか？
1.の普通の予測値
# Calculate y of training dataset
calculated_ytrain = np.ndarray.flatten(regression_model.predict(autoscaled_Xtrain))
calculated_ytrain = calculated_ytrain * ytrain.std(ddof=1) + ytrain.mean()

1.のCVでの予測値
# Estimate y in cross-validation
estimated_y_in_cv = np.ndarray.flatten(
    model_selection.cross_val_predict(regression_model, autoscaled_Xtrain, autoscaled_ytrain, cv=fold_number))
estimated_y_in_cv = estimated_y_in_cv * ytrain.std(ddof=1) + ytrain.mean()

model.predict(X,y) は、最適化したモデルにそのまま予測させる。
cross_val_predict(model,X,y) は、モデルに対して、CVでの各foldで学習＆予測させる
よって、予測値は異なる!

cross_val_predictは、CVでの各foldの予測値をまとめて返す
cross_val_scoreは、CVでの各foldの予測値それぞれから計算したスコア(R^2, 正解率)を返す
cross_val_predictから、各foldのスコアを計算→cross_val_scoreの平均値を再現することも可能

example:
test0_cross_val_predict.py

ref:
http://scikit-learn.org/stable/modules/model_evaluation.html
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html
http://scikit-learn.org/0.18/modules/cross_validation.html
https://qiita.com/nazoking@github/items/13b167283590f512d99a
https://code.i-harness.com/ja/q/2789c92
https://stackoverflow.com/questions/41458834/how-is-scikit-learn-cross-val-predict-accuracy-score-calculated
http://nakano-tomofumi.hatenablog.com/entry/2017/11/09/142828


[1a3] cross_val_predict 2.yの予測値の再現
(a) GridSearhCV
(b) splitを使った手動CV
(c) cross_val_predictの比較
として、
(a)のスコア = (b)のスコア
(b)の予測値 = (c)の予測値
を再現できた。よって、(b)か(c)で(a)を再現できる。(c)の方が簡潔。

Shufflesplitはcross_val_predictに使えない。
    (c)で再現できないだけで、(b)なら再現可能
KFold(shuffle=False)はそのまま再現可能
KFold(shuffle=True)は、別の箇所で乱数を発生させて、
random_state = ??? と代入する形であれば、再現可能
→
例の総当たり計算がやったような、CVでの予測値は計算可能と判明！

(a)のスコア
gscv.scoreは、また別の乱数を使うので再現不可能
i_best=gscv.cv_results_['params'].index(gscv.best_params_)
score0=gscv.cv_results_['split0_test_score'][i_best]
を使っていく

ref:
0613/test0.py



[1a5] DCVの結果はどう表示するべき？
https://note.mu/univprof/n/n58399e0a9471
■CalculatedY.csv ・・・ それぞれの回帰分析手法におけるモデル構築用データの目的変数の計算値
■PredictedYcv.csv ・・・ それぞれの回帰分析手法におけるモデル構築用データの
　目的変数のクロスバリデーション予測値
■PredictedY1.csv ・・・ それぞれの回帰分析手法における予測用データ1の目的変数の予測値
■PredictedY2.csv ・・・ それぞれの回帰分析手法における予測用データ2の目的変数の予測値
  ※予測用データ2は目的変数が用意されていない
■StatisticsAll.csv ・・・ それぞれの回帰分析手法におけるモデル構築用データのr^2・RMSE、
　クロスバリデーション後のr^2cv・RMSEcv、予測用データ1のr^2pred・RMSEpredの値

■StandardRegressionCoefficients.csv ・・・ 線形回帰分析手法
　(OLS, PLS, RR, LASSO, NE, LSVR)における標準回帰係数

■PredictedYdcv.csv ・・・ それぞれの回帰分析手法におけるモデル構築用データの
　目的変数のダブルクロスバリデーション予測値
 http://univprof.com/archives/16-06-12-3889388.html (省略も可能)
■StatisticsDcvAll.csv ・・・ それぞれの回帰分析手法におけるモデル構築用データの
　ダブルクロスバリデーション後のr^2dcv・RMSEdcvの値 (省略も可能)

CalculatedY.csv・PredictedYcv.csv・PredictedY1.csv・PredictedYdcv.csvのそれぞれに
対応するすべての回帰分析における目的変数の実測値と予測値とのプロット(4個×10手法=40個) 
も出力されます (プロットの省略も可能)。


CV,DCV後の予測値とは、R^2などを計算するときに使うやつ！？
sklearnのGridSearchCVでは、R^2は各fold毎にR^2を計算するので、真似は難しいか
できなくても問題なさそうだが
と、以前なら考えていた。
一応、今回のことから、真似できることは判明。
やるかどうかは別。