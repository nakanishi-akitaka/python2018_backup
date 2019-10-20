# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 23:11:53 2018

@author: Akitaka
"""
[1] 機械学習
[1a] サイトで勉強１　金子研究室
25周目: 12/10~12/16
データ解析に関するいろいろな手法・考え方のまとめ
https://datachemeng.com/summarydataanalysis/
一週間の予定
月：数学(行列計算・線形代数・統計・確率)が不安な方へ, データの前処理
火：データセットの可視化・見える化, クラスタリング, 変数選択
水：回帰分析
木：クラス分類, アンサンブル学習, 半教師あり学習 (半教師付き学習)
金：モデルの検証, モデルの適用範囲, モデルの解釈, モデルの逆解析
土：実験計画法, 時系列データ解析 (ソフトセンサーなど)
日：異常検出・異常診断, その他


[1a2] twitterから
https://twitter.com/hirokaneko226/status/1072666911429877760
Optimal Piecewise Linear Regression Algorithm with Regularisation (OPLRAreg) 
という記述子選択しながら記述子で区間に分けて、区間ごとに線形回帰モデルを構築する手法に関する論文。
SVRと同程度の性能であることを検証。
コードは↓で公開
https://github.com/KISysBio/qsar-models
「コードを公開」とのことだが、使用ライブラリ（OPLRAregが入っている）のコードは見つからない。
サンプルデータと、それを実行したノートブックは見つかったが。

"Optimal Piecewise Linear Regression Algorithm for QSAR Modelling"
Jonathan Cardoso‐Silva, George Papadatos, Lazaros G. Papageorgiou, Sophia Tsoka
https://doi.org/10.1002/minf.201800028



[1a3] SVRの誤差不感帯(εチューブ)
https://datachemeng.com/supportvectorregression/
誤差関数 h(y-f(x)) = max(0, |y-f(x)|-ε)
で、ε=2^-10, 2^-9,...,2^-1,2^0とする。つまり、ε=<1
ということは、「yを標準化」するかどうかで、εの最適値が変わってしまうのでは？
今までは、Xのみで、yは標準化してなかった！

https://github.com/hkaneko1985/gapls_gasvr
https://github.com/hkaneko1985/fastoptsvrhyperparams
これらのサンプルでは、SVRでyも標準化している
    標準化あり
    Elapsed time in hyperparameter optimization: 181.85343027114868 [sec]
    C: 8.0, Epsion: 0.015625, Gamma: 0.0009765625
    r2: 0.991063659439131
    RMSE: 54.15011242035301
    MAE: 32.61660659181073
    r2cv: 0.9626266802209518
    RMSEcv: 110.73893574328268
    MAEcv: 88.7855935003696
    r2p: 0.9613920444811777
    RMSEp: 108.40120484362481
    MAEp: 85.96520390298863
    
    標準化なし
    Elapsed time in hyperparameter optimization: 184.95222187042236 [sec]
    C: 512.0, Epsion: 0.0009765625, Gamma: 0.00390625
    r2: 0.9893003029258636
    RMSE: 59.252304784841016
    MAE: 28.23019445989608
    r2cv: 0.9459358606514938
    RMSEcv: 133.19077742822986
    MAEcv: 104.25677033742501
    r2p: 0.9495319000787483
    RMSEp: 123.93791457862005
    MAEp: 97.91936918292804

意外と変わらない？
yの値は-2000 ~ +2000まで変わっているので、無関係ということは無いハズだが

https://datachemeng.com/ordinaryleastsquares/
重回帰分析の場合は、yもXも標準化

scikit-learnのSVRのサンプルでは？
https://scikit-learn.org/stable/auto_examples/applications/plot_prediction_latency.html
yもXも標準化している！
    ただし、このサンプルで、yの標準化をなくしても、違いがいまいち見えない

他のサンプルでは、yもXも標準化してない
https://scikit-learn.org/stable/auto_examples/plot_kernel_ridge_regression.html
https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html

ソースコードやドキュメントでは、εはyのスケールに依存すると書いている
https://github.com/scikit-learn/scikit-learn/blob/55bf5d9/sklearn/svm/classes.py
    epsilon : float, optional (default=0.1)
        Epsilon parameter in the epsilon-insensitive loss function. Note
        that the value of this parameter depends on the scale of the target
        variable y. If unsure, set ``epsilon=0``.

https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html
epsilon : float, optional (default=0.1)
    Epsilon parameter in the epsilon-insensitive loss function.
    Note that the value of this parameter depends on 
    the scale of the target variable y. If unsure, set epsilon=0.

自分用のサンプルでもチェック
test0_SVR.py
    X, y = make_regression(n_samples=1000, n_features=2, n_informative=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    epsilonのみ最適化

yの標準化あり
Best parameters set found on development set:
{'epsilon': 0.0009765625, 'kernel': 'rbf'}
C:  RMSE, MAE, R^2 =  0.231,  0.043,  0.947
CV: RMSE, MAE, R^2 =  0.376,  0.095,  0.858
TST:RMSE, MAE, R^2 =  0.185,  0.141,  0.966
    ランダム分割の状況によっては、最適なεが8倍に大きくなる
Best parameters set found on development set:
{'epsilon': 0.0078125, 'kernel': 'rbf'}
C:  RMSE, MAE, R^2 =  0.214,  0.050,  0.954
CV: RMSE, MAE, R^2 =  0.415,  0.124,  0.828
TST:RMSE, MAE, R^2 =  0.249,  0.232,  0.938

yの標準化なし
Best parameters set found on development set:
{'epsilon': 0.0009765625, 'kernel': 'rbf'}
C:  RMSE, MAE, R^2 = 72.066, 56.112,  0.242
CV: RMSE, MAE, R^2 = 75.083, 59.473,  0.177
TST:RMSE, MAE, R^2 = 78.528, 62.158,  0.199
    ランダム分割の状況によっては、最適なεが最大になってしまう
    ε = 0.0009765625, ..., 1.0
Best parameters set found on development set:
{'epsilon': 1.0, 'kernel': 'rbf'}
C:  RMSE, MAE, R^2 = 85.880, 64.927,  0.168
CV: RMSE, MAE, R^2 = 88.489, 69.674,  0.116
TST:RMSE, MAE, R^2 = 100.085, 74.933,  0.117

RMSE,MAEはもともとのyの値が違うからともかくとして、R^2が大違い！
サンプル数が上の例(1000,1100)と違って、100と少ないから？

サンプル数1000個
標準化あり
Best parameters set found on development set:
{'epsilon': 0.001953125, 'kernel': 'rbf'}
C:  RMSE, MAE, R^2 =  0.025,  0.005,  0.999
CV: RMSE, MAE, R^2 =  0.044,  0.008,  0.998
TST:RMSE, MAE, R^2 =  0.108,  0.101,  0.988
17.44 seconds 

標準化なし
Best parameters set found on development set:
{'epsilon': 1.0, 'kernel': 'rbf'}
C:  RMSE, MAE, R^2 = 45.528, 23.502,  0.767
CV: RMSE, MAE, R^2 = 49.595, 27.140,  0.724
TST:RMSE, MAE, R^2 = 50.950, 26.555,  0.724
13.50 seconds 
...
Best parameters set found on development set:
{'epsilon': 0.125, 'kernel': 'rbf'}
C:  RMSE, MAE, R^2 = 10.728,  3.925,  0.898
CV: RMSE, MAE, R^2 = 12.110,  4.748,  0.870
TST:RMSE, MAE, R^2 =  9.768,  3.523,  0.907
15.69 seconds 

サンプル数10000個
標準化あり
Best parameters set found on development set:
{'epsilon': 0.0009765625, 'kernel': 'rbf'}
C:  RMSE, MAE, R^2 =  0.016,  0.001,  1.000
CV: RMSE, MAE, R^2 =  0.026,  0.002,  0.999
TST:RMSE, MAE, R^2 =  0.039,  0.019,  0.998
324.93 seconds 

標準化なし
Best parameters set found on development set:
{'epsilon': 0.125, 'kernel': 'rbf'}
C:  RMSE, MAE, R^2 = 11.678,  2.559,  0.973
CV: RMSE, MAE, R^2 = 12.942,  3.030,  0.967
TST:RMSE, MAE, R^2 = 12.242,  2.613,  0.970
1174.28 seconds

[!] データ数が増えると、標準化なしでのRMSE,MAE,R^2が、標準化ありのものに近づく
[?] epsilonが小さい理由は？オーバーフィット＝誤差が小さい→εが小さくなる、ということ？
[?] スケーリングの有無では、最適なepsilonの値が変わる程度だから、気にする必要はない？
[!] εチューブの外側の誤差が結局重要なのは、スケーリングの有無によらない
    しかし、スケーリングしないと、εチューブの中に入るサンプルの数が変わってしまう
    それならそれで、εの範囲を変えれば等価であるとは言える
    yをスケーリングすると、あとで一々逆変換する必要は出てきてしまう


kNNの場合は、n=100でも、ほぼ差なし。
標準化あり
Best parameters set found on development set:
{'n_neighbors': 3}
C:  RMSE, MAE, R^2 =  0.143,  0.102,  0.980
CV: RMSE, MAE, R^2 =  0.231,  0.166,  0.946
TST:RMSE, MAE, R^2 =  0.486,  0.330,  0.887
1.04 seconds 

標準化なし
Best parameters set found on development set:
{'n_neighbors': 3}
C:  RMSE, MAE, R^2 =  9.852,  6.283,  0.982
CV: RMSE, MAE, R^2 = 16.301, 11.872,  0.950
TST:RMSE, MAE, R^2 = 19.322, 12.213,  0.931
0.97 seconds 


[1a4] 決定係数R^2
定義式は複数あるらしい(Wikipedia)
一般的な定義は、
R^2 = 1 - Σ(y_obs - y_pred)^2/Σ(y_obs - <y_obs>)^2
[0:1]と言われることもあるが、マイナスにはなり得るのでは？（めちゃくちゃ精度が悪い時）
RMSEやMAEと同様、残差 y_obs - y_pred を基にした精度の評価なので、
残差=0 (完全一致の予測)であれば、RMSE=MAE=0のように、R^2=1となる。
標準化があってもなくても、Σ(y_obs - <y_obs>)^2で割っているから、本来はその影響は少ないハズ。
サンプルが少ない場合は、外れ値の影響を受けやすくなって（残差を二乗しているため尚更）、
R^2が小さくなっているものと考える

https://ja.wikipedia.org/wiki/%E6%B1%BA%E5%AE%9A%E4%BF%82%E6%95%B0
https://funatsu-lab.github.io/open-course-ware/basic-theory/accuracy-index/

