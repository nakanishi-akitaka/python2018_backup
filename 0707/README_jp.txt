# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 23:10:16 2018

@author: Akitaka
"""

double cross validation
  test1_dcv.py 
DCVする目的：
計算手法(SVR, LASSOなど)の決定([1,2])
回帰係数(a*x+bのa,bなど)の決定ではない([3])
ランダムに分割するので、分割の仕方によって左右される
→繰り返しが必要

ref:
[1]
クロスバリデーションでは予測性能を評価できない！
ダブルクロスバリデーション(クロスモデルバリデーション)のススメ
http://univprof.com/archives/16-06-12-3889388.html

理由：
クロスバリデーション(CV)で学習したモデル　＝　教師データセットに特化したモデル

問題：
[OK] CVでハイパーパラメータ決定
[NG] CVで予測性能の推定

解決：ダブルクロスバリデーション(DCV)
クロスバリデーションを入れ子構造にして2回使う

具体例：
1.分割　データセット＝サブデータセットA＋...B
2.各サブデータでCV
3.各サブデータの予測値を得る
4.R^2_DCVなどを得る　だいたいはR^2_CVより悪い
5.何度もDCVを行って、r2, RMSE, 正解率の分布を見る
  データのランダム分割によって、CV, DCVの結果は変わるため
6.別の計算手法でもDCV(LASSO, SVMなど)
7.どの計算手法を採用するか決定する
  ※後述の論文とは違う
  たとえば、PLSで100回行った場合より、SVRで100回行った場合のほうがR^2CDVのばらつきが小さいなら
  そのデータセットでは、SVRの方が回帰分析手法として安定している、といえる



[2]
ダブルクロスバリデーション(モデルクロスバリデーション)でテストデータいらず
～サンプルが少ないときのモデル検証～
https://datachemeng.com/doublecrossvalidation/
データ分割はLOOかN-fold(N=2,5)
N-foldの場合、ランダム性があるので、何回もやる



[3]
P. Filzmoser, B. Liebmann, K. Varmuza,
"Repeated double cross validation"
J. Chemom. 23, 160–171, 2009.
https://doi.org/10.1002/cem.1225
[2]の説明の元になった論文。内容はほぼ同様なので省略
ここでは、ランダム＆N-foldなので、n(rep)回繰り返しやっている
繰り返しの後は？
上の説明では、PLSやSVRなどの選択に使う
この論文では、PLSの回帰係数の分布から、
最も出現頻度の高いa^optを最終的な値としている
※どの係数a^optもn(rep)個ある
※回帰係数を決めるのは、ハイパーパラメータを決めるのとは違う
SVMなら、wが回帰係数、ハイパーパラメータはC,ε,γ
SVMでDCVをやるのは大変だろうか？



[4]
クロスバリデーションとダブルクロスバリデーションの違い
http://chemstudentlab.com/2017/10/18/クロスバリデーションとダブルクロスバリデーシ/
データ分割はLOO



[5] 自分の考察
例えば DS→DS1, DS2, DS3と分けて、
DS2,DS3でin-CV = ハイパーパラメータ決定
DS2+DS3でtrain(with最適化したハイパーパラメータ), DS1でtestする→スコア1
同様に
DS1,DS3でin-CV → ...DS2でtestする→スコア2
DS1,DS2でin-CV → ...DS3でtestする→スコア3

スコア1,2,3の平均を予測性能とする

ref:模式図
http://weka.8497.n7.nabble.com/file/n35703/slide_81.jpg
http://weka.8497.n7.nabble.com/Nested-crossvalidation-results-meaning-td28767.html
https://sebastianraschka.com/faq/docs/evaluate-a-model.html




KFoldを使えば、たぶん行ける
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
  test4.py
train_index, test_index = KFold.split(X)で、インデックスのリスト(IL)を取得する
例：4分割の場合
DS1+DS2+DS3のIL, DS4のIL
DS1+DS2+DS4のIL, DS3のIL
DS1+DS3+DS4のIL, DS2のIL
DS2+DS3+D43のIL, DS1のIL

pythonコードでは以下の通り
kf = KFold(n_splits=4, shuffle=True)
for train_index, test_index in kf.split(X):
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]

よって、ここでさらにX_train, y_trainを用いてCVを行えばいいハズ
mod = SVR() 
param_grid = [{...}]
cv_kf = KFold(n_splits=4, shuffle=True)
gscv = GridSearchCV(mod, param_grid, cv=cv_kf, scoring='r2')
gscv.fit(X_train, y_train)
