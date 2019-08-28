# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 10:06:59 2018

@author: Akitaka
"""


[1a2] プログラムを組む
SVM+OCSVM+DCVの、kNNバージョン
OCSVMはADだけだが、kNNはAD＋信頼度
example:
test0_kNN_AD_DCV_clf.py

信頼度 (標準偏差 ver.)を追加

信頼度 (平均 ver.1)
y_reli = np.absolute(gscv.predict_proba(X_test)[:,1]-0.5)+0.5

信頼度 (平均 ver.2 = 平均 ver.1)
y_reli = np.absolute(np.mean(y_train[neigh.kneighbors(X_test)[1]], axis=1)
    -0.5)+0.5

信頼度 (標準偏差 ver. =/= 平均 ver.1, 2)
y_reli = 1 - np.std(y_train[neigh.kneighbors(X_test)[1]], axis=1)


07/27追記
> k-NNでの信頼度の計算は？(平均と標準偏差、多数決の割合など)
> 信頼度と適用範囲について
>        信頼度＝1と0の割合(分類), 標準偏差(回帰)　→　どっちも標準偏差！
としたが、標準偏差でも表すことができる、というだけ。
割合は平均値なので数値は異なる。


[1a3] さらに回帰ver.も作る
example:
test0_kNN_AD_DCV_rgr.py

回帰では、信頼度ではなく、不確かさ・標準偏差とする。
# Standard Deviation (= Uncertainty <-> Reliability)
y_reli = np.std(y_train[neigh.kneighbors(X_test)[1]], axis=1)



[1c1] 水素化物のTc予測プログラム
example:
test1_to_csv.py
example:
test2_Tc_SVM_OCSVM_DCV.py

昨日：テスト計算なので、探索範囲を小さくしている
今日：探索範囲を推奨値にする

結果

Search range
c =  0.03125  ...  1024.0
e =  0.0009765625  ...  1.0
g =  9.5367431640625e-07  ...  1024.0

Best parameters set found on development set:
{'svr__C': 256.0, 'svr__epsilon': 1.0, 'svr__gamma': 4.0, 'svr__kernel': 'rbf'}

Grid scores on development set:
train data: RMSE, MAE, RMSE/MAE, R^2 = 15.777, 7.620, 2.070, 0.839

DCV
  ave, std of accuracy of inner CV: 0.448 (+/-0.078)
  ...
  ave, std of accuracy of inner CV: 0.121 (+/-0.176)
1524.95 seconds 

結果にも出力 AD == 1 のみを抽出して出力
Tc max CaH6,150,80,1 他省略
H3Sすら、ADから外れた！
流石に何かおかしいのでは？


[1c3] DCVの分割数を変える
n_splits = 2
上記の通り

n_splits = 3
  ave, std of accuracy of inner CV: 0.331 (+/-0.081)
  ...
  ave, std of accuracy of inner CV: 0.550 (+/-0.120)
897.16 seconds 

n_splits = 5
Predicted Tc is written in file test2.csv
  ave, std of accuracy of inner CV: 0.453 (+/-0.241)
  ...
  ave, std of accuracy of inner CV: 0.589 (+/-0.114)
3957.51 seconds 

注意：探索範囲は少し減らしてある
結果はn_splits=2よりも明らかによくなっている


[1c4] OCSVMのハイパーパラメータをかえてみる
データベースにあったサンプル数から明らかに減り過ぎている
nu=0.10が大きすぎた？
3σ法から言えば、nu=0.003
0726/test0_SVM_OCSVM_contour.py 
でテストした
nuを減らすとADが増え、nuを増やすとADが減る
nu=0.003だと、ほとんどのサンプルがADの中に入る

nu=0.003でテスト
※DCVは省略

結果
H3SもADに入るようになった
しかし、依然として少ない
formula,P,Tc,AD
H3S,250,182,1
H3S,300,178,1
LiH6,300,80,1
以下省略

nu=0.00003にまで減らすと、逆に(？)全部ADの外

？γはどうやって決める？
[1] グラム行列を最大にする
[2] SVMと同じ値
と意見が一致しない

ref:
[1] https://datachemeng.com/ocsvm/
[2] http://univprof.com/archives/17-01-28-11535179.html


[1c5] ADを無視して、とりあえず一番高いTc予測がどうなるかを見てみる
Top10は以下の通り

formula,P,Tc,AD
VH3,200,299,-1
CrH3,200,297,-1
TiH3,200,296,-1
MnH3,200,291,-1
ScH3,200,288,-1
CrH3,150,284,-1
VH3,150,284,-1
FeH3,200,281,-1
TiH3,150,280,-1
MnH3,150,279,-1

Tc > 200 K だけでも90ぐらいはある。


[1c6] SVM以外もやってみる k-NN
example:
test2_Tc_kNN_AD_DCV.py

Search range
k =  3  ...  10

Best parameters set found on development set:
{'model__n_neighbors': 3}

Grid scores on development set:
train data: RMSE, MAE, RMSE/MAE, R^2 = 14.488, 8.648, 1.675, 0.864

DCV
Predicted Tc is written in file test2.csv
  ave, std of accuracy of inner CV: 0.028 (+/-0.355)
  ...
  ave, std of accuracy of inner CV: 0.181 (+/-0.142)
26.87 seconds 

Top10
formula,P,Tc,Std,AD
ScH3,250,193,65.19,1
VH3,200,193,74.12,1
VH3,250,193,65.19,1
CaH3,250,193,65.19,1
H3S,250,193,65.19,-1
CrH3,250,193,62.09,1
H3Cl,250,193,65.19,-1
CrH3,200,193,74.12,1
KH3,250,193,65.19,-1
TiH3,250,193,65.19,1
AD内に、200K近いものがある！
H3SがまたADの外に……


[1d1] SVM+OCSVM+ダブルクロスバリデーション/二重交差検証/DCV プログラム
ここで一旦開発は中止。test0_...とリネーム
ref:
0713, 0716, 0717,0718, 0725, 0726
http://univprof.com/archives/16-06-12-3889388.html
https://datachemeng.com/doublecrossvalidation/

example:
test0_SVM_OCSVM_DCV_rgr.py
test0_SVM_OCSVM_DCV_clf.py
