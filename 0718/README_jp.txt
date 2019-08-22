# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 10:38:48 2018

@author: Akitaka
"""
[1d] SVM+OCSVM
ref:0713, 0716, 0717
example:
test1_SVM_OCSVM.pyをアップデート




[1e] Tcへの応用
example:
test2_Tc_SVM.py

[1e1]
Tcファイルの読み込み部分だけ、0521test3.pyを参考にして、残りはtest0_rgr.pyを基に作成
方法
水素化物Tcデータベース→train, testに分割
探索は無し

ref:
~/python_work/2018/0521/test2.py tc.csvファイルで化学式を原子番号と個数に変換
~/python_work/2018/0521/test3.py SVMでtc予測



Kfold(shuffle=True)とShuffleSplit の違い
1.train, testのサイズ:
    KFoldは常に1:k-1(k=分割数), ShuffleSplitは(分割数とは無関係に)任意の変更が可能
    つまり、trainにもtestにも使わないデータもあり得る。データが多すぎる時に有効
    n_splits:繰り返しの数(デフォ10), test_size:testデータのサイズ(デフォ0.1)
2.CVの各イテレーションで使うデータセット:
    KFold：どれも必ず一度だけtestとして使う。最初にランダムに分割しても
    ShuffleSplit:毎回ランダムに抽出するので、何度もtestに使うもの、一度も使わないものがありえる
example:
test3_shuffle.py

















[1d] SVM+OCSVM
ref:0713, 0716, 0717
example:
test1_SVM_OCSVM.pyをアップデート




[1e] Tcへの応用
example:
test2_Tc_SVM.py

[1e1]
Tcファイルの読み込み部分だけ、0521test3.pyを参考にして、残りはtest0_rgr.pyを基に作成
方法
水素化物Tcデータベース→train, testに分割
探索は無し

ref:
~/python_work/2018/0521/test2.py tc.csvファイルで化学式を原子番号と個数に変換
~/python_work/2018/0521/test3.py SVMでtc予測


結果
Search range
c =  0.125  ...  512.0
e =  0.03125  ...  0.5
g =  0.0009765625  ...  16.0
Best parameters set found on development set:
{'C': 512.0, 'epsilon': 0.5, 'gamma': 1.0, 'kernel': 'rbf'}

train data: RMSE, MAE, RMSE/MAE, R^2 = 5.303, 2.929, 1.810, 0.980
test  data: RMSE, MAE, RMSE/MAE, R^2 = 34.556, 21.251, 1.626, 0.432
82.48 seconds 


[1e2]
以前の研究室ミーティングで発表したデータは？
0426test1, 0427test1, 0511test1のどれか
0511は発表前に軽く計算しただけなのか、メールに詳細を書いていない
ファイルの差分をとると、 0427と同様

0427は多少
https://mail.google.com/mail/u/0/#sent/LXphbRLrghxkrJlVCBvNsNFkgTbLZZcQmQjcShDJmPg
メインは、スケーリングをMinMax, Standardのどちらにするかの比較

0426が重要
cvの数、cv=KFold, Shufflesplitなどの違いに言及
Shufflesplitがベストと結論
また、ShuffleSplitとscalingを併用することで性能が向上したとも書いてある
https://mail.google.com/mail/u/0/#sent/RdDgqcJHpWcvcDjPgjkjXHLgLnDfdlQzrnZXHZlrxmfB
以下、引用
    [1] 機械学習
    総当たり計算 0426/test1.py
    [1a] cvの分配数を変えると？
    ※以下、パラメータ最適化のみの場合
    cv, R^2,    b-score, Tc>120K
     3, 0.931,  0.0139,  3
     4, 0.931, -0.0378,  3
     5, 0.702, -0.0488,  30
     6, 0.753, -0.0791,  180
     7, 0.966, -0.1084,  8
     8, 0.949, -0.2636,  3
     9, 0.839, -0.1141,  1
    10, 0.832, -0.7814,  0
    
    [1b] ShuffleSplitを使い、分配数を変えると？
    ※Tc>120Kの数は省略、test_size=0.1で固定
     cv = ShuffleSplit(n_splits=icv, test_size=0.1, random_state=0)
    icv, R^2, b-score,
     3 0.966 0.412
     4 0.966 0.488
     5 0.959 0.523
     6 0.966 0.565
     7 0.966 0.575
     8 0.966 0.597
     9 0.966 0.598
    10 0.966 0.577
    best scoreが大幅に改善された！
    また、Tc>120Kはどれも8個
    つまり、訓練データファイル内で偏りがあったことが問題だった
    それでもR^2とbest scoreが離れているが
    ※デフォルトでn_splits=10, test_size=0.1
    
    [1c] 分割方法について調べる
    KFoldだと、単純に順番通りに分割する
    　データの偏りがあった時、各fold間で性能が大きく異なる
    StratifiedKFold（デフォルト）では、データないの偏りをなるべく維持しようとする
    　少数データの数が少ないと分割ができない場合もある
    ShuffleSplitは単純にランダムサンプリングを行う
    LeaveOneOutは1つだけを取り出す
    　データ数が少ないときに有効
    
    http://scikit-learn.org/0.18/modules/cross_validation.html
    > As a general rule, most authors, and empirical evidence,
    > suggest that 5- or 10- fold cross validation should be preferred to LOO.
    LeaveOneOutは一般的にはあまり使わない？
    > ShuffleSplit is thus a good alternative to KFold cross validation
    > that allows a finer control on the number of iterations
    > and the proportion of samples on each side of the train / test split.
    ShuffleSplitが良いらしい
    
    <ref>
    scikit learn の Kfold, StratifiedKFold, ShuffleSplit の違い
    http://nakano-tomofumi.hatenablog.com/entry/2018/01/15/172427
    Machine Learning with Scikit Learn (Part II)
    http://aidiary.hatenablog.com/entry/20150826/1440596779
    3.1. Cross-validation: evaluating estimator performance
    http://scikit-learn.org/0.18/modules/cross_validation.html
    【翻訳】scikit-learn 0.18 User Guide 3.1. クロスバリデーション：推定器の成果を評価する
    https://qiita.com/nazoking@github/items/13b167283590f512d99a
    活動記録　2018/04/11
    </ref>
    
    [1d] 他の分割方法を試す LeaveOneOut
    →時間がかかったので中止
    
    [1e] 他の分割方法を試す StratifiedShuffleSplit
    →二値分類でないとダメ
    
    [TODO]->[DONE] cvの分配方法や、データの順序は関係ある？
    とりあえずは、ShuffleSplitを使う
    
    [1f] ShuffleSplitとscalingを併用する
    ->性能向上！
    Cとγの探索範囲は0.001~9000なので、最適解＝端を回避できた
    # SVR with GridSearched hyper parameters after MinMaxScaler
    best_score  : 0.8053063569094222
    best_params : {'svr__C': 3000, 'svr__gamma': 200, 'svr__kernel': 'rbf'}
    learning   score: RMSE, MAE, RMSE/MAE, R^2 = 7.766, 3.974, 1.954, 0.961
    Tc>150K, 100個
    
    # SVR with GridSearched hyper parameters after StandardScaler
    best_score  : 0.732353201310859
    best_params : {'svr__C': 6000, 'svr__gamma': 5, 'svr__kernel': 'rbf'}
    learning   score: RMSE, MAE, RMSE/MAE, R^2 = 8.948, 4.518, 1.981, 0.948
    Tc>150K, 553個
    
    [1g] 説明変数を増やす
    原子番号　原子数　圧力
    →族　周期　電気陰性度　最大酸化数　最小酸化数　原子量　メンデレーエフ数　融点　熱伝導率
    ※予測の時に使えないもの、空間群、体積(希ガスや液体の元素にはない)などは説明変数から省いた
    # SVR with GridSearched hyper parameters after MinMaxScaler
    search score
    best_score  : 0.8548804762530285
    best_params : {'svr__C': 500, 'svr__gamma': 30, 'svr__kernel': 'rbf'}
    learning   score: RMSE, MAE, RMSE/MAE, R^2 = 7.423, 4.163, 1.783, 0.965
    Ca1H6 P = 100 GPa Tc = 150.0 K
    Ca1H6 P = 150 GPa Tc = 163.1 K
    Ca1H6 P = 200 GPa Tc = 150.0 K
    As1H8 P = 450 GPa Tc = 151.3 K
    As1H8 P = 500 GPa Tc = 151.7 K
    Tc>150K, 5個
    best score とR^2について
    grid毎のscoreを表示させたところ
    0.855 (+/-0.179) for {'svr__C': 500, 'svr__gamma': 30, 'svr__kernel': 'rbf'}
    分散が大きくなるせいでR^2との違いが大きいと分かる
    
    [1h] +PCAも追加
    # SVR with GridSearched hyper parameters after MinMaxScaler
    search score
    best_score  : 0.8665341432084145
    best_params : {'pca__n_components': 6, 'svr__C': 200, 'svr__gamma':
    100, 'svr__kernel': 'rbf'}
    learning   score: RMSE, MAE, RMSE/MAE, R^2 = 8.784, 5.203, 1.688, 0.952
    Ca1H5 P = 250 GPa Tc = 150.2 K
    Ca1H6 P = 050 GPa Tc = 152.0 K
    Ca1H6 P = 100 GPa Tc = 160.2 K
    Ca1H6 P = 150 GPa Tc = 163.1 K
    Ca1H6 P = 200 GPa Tc = 160.2 K
    Ca1H6 P = 250 GPa Tc = 152.0 K
    Ca1H7 P = 050 GPa Tc = 150.2 K

http://scikit-learn.org/0.18/modules/cross_validation.html
> However, if the learning curve is steep for the training size in question,
> then 5- or 10- fold cross validation can overestimate the generalization error.
汎化誤差を過大評価する危険性に注意

Kfold(shuffle=True)とShuffleSplit の違い
1.train, testのサイズ:
    KFoldは常に1:k-1(k=分割数), ShuffleSplitは(分割数とは無関係に)任意の変更が可能
    つまり、trainにもtestにも使わないデータもあり得る。データが多すぎる時に有効
    n_splits:繰り返しの数(デフォ10), test_size:testデータのサイズ(デフォ0.1)
2.CVの各イテレーションで使うデータセット:
    KFold：どれも必ず一度だけtestとして使う。最初にランダムに分割しても
    ShuffleSplit:毎回ランダムに抽出するので、何度もtestに使うもの、一度も使わないものがありえる
example:
test3_shuffle.py

ref:
http://nakano-tomofumi.hatenablog.com/entry/2018/01/15/172427
https://stackoverflow.com/questions/34731421/whats-the-difference-between-kfold-and-shufflesplit-cv
http://aidiary.hatenablog.com/entry/20150826/1440596779
https://qiita.com/nazoking@github/items/13b167283590f512d99a
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html
http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

[1e3] Shufflesplitにしてみた場合
Search range
c =  1.0  ...  512.0
e =  0.03125  ...  0.5
g =  0.0009765625  ...  16.0

Best parameters set found on development set:

{'C': 256.0, 'epsilon': 0.03125, 'gamma': 0.25, 'kernel': 'rbf'}

train data: RMSE, MAE, RMSE/MAE, R^2 = 5.191, 2.432, 2.134, 0.983
test  data: RMSE, MAE, RMSE/MAE, R^2 = 27.367, 15.695, 1.744, 0.396
103.36 seconds 
一見、大差ないがtest dataのyy-plotは異なる
ランダム性に左右されるので、特徴は書かない
