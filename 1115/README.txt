# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:38:00 2018

@author: Akitaka
"""

[1b] 水素化物Tc計算
[1b1] 考察
昨日より最適化したkNNのハイパーパラメータについて
6.5+密度のみが{'n_neighbors': 5}となり、他は{'n_neighbors': 1}
n_neighbors=1って機械学習としてほぼ無意味なのでは？
色々検索したが、k=1が特別ダメという記述はない
    k最近傍法でなく、最近傍法という名前があるらしい

ref:
https://ja.wikipedia.org/wiki/K%E8%BF%91%E5%82%8D%E6%B3%95
https://qiita.com/yshi12/items/26771139672d40a0be32
https://qiita.com/NoriakiOshita/items/698056cb74819624461f
https://www.slideshare.net/moa108/5-kknn
https://dev.classmethod.jp/machine-learning/2017ad_20171218_knn/
https://blog.amedama.jp/entry/2017/03/18/140238
http://ibis.t.u-tokyo.ac.jp/suzuki/lecture/2015/dataanalysis/L10.pdf

注意：電気陰性度、密度、熱伝導率は、一部の原子のデータが無かったため、適当な値に置き換えた
特に、電気陰性度はなぜか複素数になっているものがある！
ValueError: could not convert string to float: '(0.6950206606975026+0j)'
昨日はとりあえずabs(...)で処理したが、マイナスがプラスになってしまう
色々試行錯誤したものの、うまくいかないので、電気陰性度は使用しないことにした


[1b2] 説明変数を増やして再計算
方法は、全部試していく
予測の範囲を少し絞る
    X_n + H_mで、n>2はあまり高くならない -> n= 1~2
    P=100~300

ref
summary/0705test1_Tc_SVM_make_parameters.py
summary/0705test3_atomic_data.py
https://arxiv.org/abs/1803.10260

！scalingを最初にやったため、圧力もスケーリングされてしまっていたことに気付く
仕様変更：圧力部分のみ保存する

結果
RR
C:  RMSE, MAE, R^2 = 29.933, 20.359, 0.730
CV: RMSE, MAE, R^2 = 34.271, 23.063, 0.646
P:  RMSE, MAE, R^2 = 89.566, 71.623, 0.000
DCV:RMSE, MAE, R^2 = 34.336, 23.335, 0.645 (ave)
DCV:RMSE, MAE, R^2 =  0.879,  0.581, 0.018 (std)
rnd:RMSE, MAE, R^2 = 55.151, 40.756, 0.084 (ave)
rnd:RMSE, MAE, R^2 =  0.551,  0.593, 0.018 (std)

formula,P,Tc,AD
YH10,300,239,1
YH10,250,237,1
YH10,200,235,1
YH10,150,233,1
YH10,100,231,1
YH9,300,226,1
YH9,250,224,1
YH9,200,222,1
YH9,150,220,1
YH9,100,218,1

EN
警告文多発
C:\Users\Akitaka\Anaconda3\lib\site-packages\sklearn\linear_model\coordinate_descent.py:
491: ConvergenceWarning: Objective did not converge.
You might want to increase the number of iterations.
Fitting data with very small alpha may cause precision problems.
ConvergenceWarning)

C, CVなどのは警告文が大量に出たせいで流れてしまった
DCV:RMSE, MAE, R^2 = 35.602, 24.216, 0.618 (ave)
DCV:RMSE, MAE, R^2 =  0.711,  0.295, 0.015 (std)
rnd:RMSE, MAE, R^2 = 56.492, 41.710, 0.038 (ave)
rnd:RMSE, MAE, R^2 =  0.487,  0.290, 0.017 (std)
199.03 seconds

formula,P,Tc,AD
YH10,300,216,1
YH10,250,215,1
YH10,200,214,1
YH10,150,213,1
YH10,100,212,1
YH9,300,205,1
YH9,250,204,1
YH9,200,203,1
YH9,150,202,1
YH9,100,201,1



LASSO
警告文同様
DCV:RMSE, MAE, R^2 = 35.520, 24.401, 0.620 (ave)
DCV:RMSE, MAE, R^2 = 0.917, 0.687, 0.020 (std)
RMSE: 56.759 (+/-0.295)
rnd:RMSE, MAE, R^2 = 56.759, 41.902, 0.029 (ave)
rnd:RMSE, MAE, R^2 = 0.295, 0.226, 0.010 (std)
32.66 seconds

formula,P,Tc,AD
YH10,300,216,1
YH10,250,215,1
YH10,200,214,1
YH10,150,213,1
YH10,100,212,1
YH9,300,205,1
YH9,250,204,1
YH9,200,203,1
YH9,150,202,1
YH9,100,201,1


kNN
{'n_neighbors': 1}
C:  RMSE, MAE, R^2 =  8.097,  4.126, 0.980
CV: RMSE, MAE, R^2 = 28.952, 17.053, 0.747
P:  RMSE, MAE, R^2 = 73.495, 51.218, 0.000
DCV:RMSE, MAE, R^2 = 25.630, 14.964, 0.801 (ave)
DCV:RMSE, MAE, R^2 =  1.975,  0.927, 0.031 (std)
rnd:RMSE, MAE, R^2 = 51.154, 37.805, 0.211 (ave)
rnd:RMSE, MAE, R^2 =  1.039,  0.869, 0.032 (std)
9.48 seconds

formula,P,Tc,AD
YH9,300,267,1
YH9,150,267,1
YH9,100,267,1
YH8,300,267,1
YH8,250,267,1
YH8,200,267,1
YH8,150,267,1
YH8,100,267,1
YH9,250,267,1
YH9,200,267,1


RF
{'max_features': 0.6000000000000001}
C:  RMSE, MAE, R^2 =  9.942,  6.562, 0.970
CV: RMSE, MAE, R^2 = 19.800, 13.227, 0.882
P:  RMSE, MAE, R^2 = 79.227, 61.328, 0.000
DCV:RMSE, MAE, R^2 = 23.488, 14.482, 0.833 (ave)
DCV:RMSE, MAE, R^2 =  1.502,  0.733, 0.021 (std)
rnd:RMSE, MAE, R^2 = 40.772, 28.393, 0.498 (ave)
rnd:RMSE, MAE, R^2 =  1.632,  0.879, 0.040 (std)
106.44 seconds

formula,P,Tc,AD
YH9,200,261,1
YH9,100,261,1
YH9,250,261,1
YH9,150,261,1
YH6,100,258,1
YH10,250,258,1
YH10,200,258,1
YH10,150,258,1
YH10,100,258,1
YH6,150,257,1

！？偶然の相関が大きくないか？他のよりは大きい


GB
{'n_estimators': 200}
C:  RMSE, MAE, R^2 =  8.840,  6.188, 0.976
CV: RMSE, MAE, R^2 = 24.723, 14.516, 0.816
P:  RMSE, MAE, R^2 = 82.734, 59.040, 0.000
DCV:RMSE, MAE, R^2 = 22.881, 13.954, 0.841 (ave)
DCV:RMSE, MAE, R^2 =  2.238,  1.108, 0.031 (std)
rnd:RMSE, MAE, R^2 = 54.249, 40.128, 0.113 (ave)
rnd:RMSE, MAE, R^2 =  0.553,  0.432, 0.018 (std)
253.42 seconds

formula,P,Tc,AD
YH9,100,256,1
YH9,150,256,1
YH9,200,254,1
YH9,250,253,1
YH10,200,252,1
YH9,300,251,1
YH10,100,251,1
YH10,300,251,1
YH6,100,251,1
YH10,150,250,1


SVR
Best parameters set found on development set:
{'C': 256.0, 'epsilon': 1.0, 'gamma': 1.0}
C:  RMSE, MAE, R^2 = 11.113,  6.433, 0.963
CV: RMSE, MAE, R^2 = 27.238, 16.166, 0.776
P:  RMSE, MAE, R^2 = 55.526, 52.442, 0.000
DCV:RMSE, MAE, R^2 = 25.930, 16.193, 0.796 (ave)
DCV:RMSE, MAE, R^2 =  2.217,  0.960, 0.035 (std)
rnd:RMSE, MAE, R^2 = 51.648, 29.654, 0.195 (ave)
rnd:RMSE, MAE, R^2 =  2.358,  2.677, 0.074 (std)
1684.07 seconds

formula,P,Tc,AD
YH9,300,275,1
YH9,250,272,1
YH9,200,265,1
YH10,300,262,1
YH9,150,254,1
YH6,100,250,1
YH6,150,249,1
YH10,250,245,1
YH9,100,238,1
YH6,200,237,1


GPR
C:  RMSE, MAE, R^2 = 5.875, 3.837, 0.990
35.26 seconds

formula,P,Tc,std,AD
YH9,150,260,1.1185797705685159,1
YH10,250,254,0.5240784322265132,1
YH10,300,244,0.5240784322265132,1
LaH10,250,222,0.5240784322265132,1
LaH10,300,205,0.5240784322265132,1
H3S,200,197,1.1185797705685159,1
H3S,250,183,9.996535917871775e-06,1
H3S,300,178,9.996535917871775e-06,1
ScH7,300,168,9.996535917871775e-06,1
CaH6,150,167,1.1185797705685159,1




[1b3] PLSの追加
説明変数を増やしたのでやってみる
重要そうでない理由は昨日書いた通りだが、一方で、PLSを重要視しているのもあるので

https://ja.wikipedia.org/wiki/部分的最小二乗回帰
https://datachemeng.com/partialleastsquares/
https://note.mu/univprof/n/n7d9eb3ce2c74

PLS
{'n_components': 23}
C:  RMSE, MAE, R^2 = 30.611, 21.085, 0.718
CV: RMSE, MAE, R^2 = 35.689, 24.289, 0.616
P:  RMSE, MAE, R^2 = 98.638, 77.701, 0.000
DCV:RMSE, MAE, R^2 = 38.003, 25.658, 0.564 (ave)
DCV:RMSE, MAE, R^2 =  2.020,  0.934, 0.048 (std)
rnd:RMSE, MAE, R^2 = 57.200, 42.156, 0.014 (ave)
rnd:RMSE, MAE, R^2 =  0.341,  0.223, 0.012 (std)
167.24 seconds

formula,P,Tc,AD
YH10,300,231,1
YH10,250,229,1
YH10,200,227,1
YH10,150,225,1
YH10,100,223,1
YH9,300,220,1
YH9,250,218,1
YH9,200,215,1
YH9,150,213,1
YH9,100,211,1


[1b4] 考察・議論
学習用データにあった high-Tc 物質をそのまま高いTc候補として出している？
ほぼほぼYH10, YH9がtop10に来る
tc.csvより
  YH9  ,  194,      150,        ,           , 0.10,      267.2,
https://doi.org/10.1103/PhysRevLett.119.107001
  YH9  ,  194,      150,        ,           , 0.13,      253.2,
https://doi.org/10.1103/PhysRevLett.119.107001
  YH10 ,  225,      400,        ,           , 0.10,      310.2,
https://doi.org/10.1103/PhysRevLett.119.107001
  YH10 ,  225,      400,        ,           , 0.13,      287.2,
https://doi.org/10.1103/PhysRevLett.119.107001
  YH10 ,  229,      250,    2.58,       1282, 0.10,        265,
https://doi.org/10.1073/pnas.1704505114
  YH10 ,  229,      250,    2.58,       1282, 0.13,        244,
https://doi.org/10.1073/pnas.1704505114
  YH10 ,  229,      300,    2.06,       1511, 0.10,        255,
https://doi.org/10.1073/pnas.1704505114
  YH10 ,  229,      300,    2.06,       1511, 0.13,        233,
https://doi.org/10.1073/pnas.1704505114
→
trainにあるかどうかを判定できないのか？

説明変数を増やした効果は？
※どちらも、更新したデータベースで計算している
ランダムフォレストで比較

==> 1113/Tc_RF_AD_DCV.csv <==
formula,P,Tc,AD
RbH9,150,235,1
SrH9,150,235,1
ZrH9,150,235,1
YH9,150,235,1
H9Ru,150,216,1
NbH9,150,216,1
TcH9,150,216,1
MoH9,150,216,1
AgH9,150,212,1

==> 1115/Tc_RF.csv <==
formula,P,Tc,AD
YH9,200,261,1
YH9,100,261,1
YH9,250,261,1
YH9,150,261,1
YH6,100,258,1
YH10,250,258,1
YH10,200,258,1
YH10,150,258,1
YH10,100,258,1

あきらかに候補となる物質が異なる！
元々のデータベースに合った物質の再現率の問題かもしれない？
1115/Tc_RF.csvでは？
RbH9 AD外
SrH9 AD外
ZrH9 AD外
YH9  上記の通り
H9Ru AD外
NbH9 AD外
TcH9 AD外
MoH9 AD外
！再現率ではなく、適用範囲(AD)の問題だった！
説明変数が変わったことによって、ADも変わってしまった

==> 1113/Tc_RF_AD_DCV.txt <==
Best parameters set found on development set:
{'model__max_features': 0.30000000000000004}
C:  RMSE, MAE, R^2 = 11.193,  7.720, 0.962
CV: RMSE, MAE, R^2 = 24.986, 16.730, 0.812
P:  RMSE, MAE, R^2 = 51.537, 42.050, 0.000
DCV:RMSE, MAE, R^2 = 29.303 (+/-1.209), 18.499 (+/-0.646), 0.741 (+/-0.021)
rnd:RMSE, MAE, R^2 = 41.138 (+/-1.067), 28.698 (+/-0.750), 0.490 (+/-0.026)
116.43 seconds

==> 1115/Tc_RF.txt <==
Best parameters set found on development set:
{'max_features': 0.6000000000000001}
C:  RMSE, MAE, R^2 =  9.942,  6.562, 0.970
CV: RMSE, MAE, R^2 = 19.800, 13.227, 0.882
P:  RMSE, MAE, R^2 = 79.227, 61.328, 0.000
DCV:RMSE, MAE, R^2 = 23.488, 14.482, 0.833 (ave)
DCV:RMSE, MAE, R^2 =  1.502,  0.733, 0.021 (std)
rnd:RMSE, MAE, R^2 = 40.772, 28.393, 0.498 (ave)
rnd:RMSE, MAE, R^2 =  1.632,  0.879, 0.040 (std)
106.44 seconds

RMSE, MAEは5程度、r^2は0.1ぐらい改善されている
手間がかかる割にはこの程度？
あるいは、改善するといっても、たいていはこの程度とか？




[1c] 考察
構造最適化の説明変数をfingerprint以外でやる？

思いついたこと
外部ポテンシャル　→　電荷密度　→　物理量がすべて1:1に対応するのだから、
外部ポテンシャルでもいいのでは？

格子定数a,b,c,α,β,γによって、格子の形が変わるのはどう対応するか？
xyz座標ではなく、格子ベクトル座標での記述であれば、問題ない
原子位置のみを格子ベクトル座標で表す場合は、a,b,c,α,β,γの情報が反映されないのでNG！
一方、ポテンシャルであれば空間に分布するものなので、OK。例えば以下のようにポテンシャルが変化する。
    a,b,cが大きい　→　原子間においてポテンシャルが少ない
    a,b,cが小さい　→　原子間においてポテンシャルが大きい
また、こうすることで、a,b,c,α,β,γに関係なく、説明変数が一定になる
ただし、どの程度、空間のメッシュをとるかという部分はパラメータになる
画像の解像度みたいなものともいえるが

どの物理量であっても、原理的には、外部ポテンシャルが分かればいい、という保証がある
原子の質量、電気陰性度、熱伝導率などをあれこれ組み合わせる場合、何が関係するのかは不透明

密度汎関数理論にでてくる、ユニバーサルな汎関数(交換相互作用と電子相関をあらわす)を導出できたことになる？
    密度ではなく、外部ポテンシャルに対する汎関数にはなるが
もし導出できれば、「どんな物質にでも」応用できるハズ
    もっとも、「学習用データ」がどんな物質でも精確に交換相関エネルギーを計算できていることが前提
ref:ユニバーサルな汎関数について
http://www.op.titech.ac.jp/polymer/lab/sando/Conf_04/SPSJ_OptoElectro.pdf

一方で、問題点
説明変数が大幅に増える。計算が重い。
増えること自体は悪くない。十分な情報量さえあればいい。
原子番号、原子の位置、格子定数があれば、一応トータルエネルギーは決まる。
それらに比べれば、説明変数は増える。
