# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 23:57:38 2018

@author: Akitaka
"""

[1a2] ピックアップ
ランダムフォレスト(Random Forest, RF)や決定木(Decision Tree, DT)で構築したモデルを逆解析するときは気をつけよう！
https://datachemeng.com/inverseanalysisrfdt/
モデルの逆解析の目的の１つ
1.y の値をなるべく大きく (または小さく) する、もしくは y を既存の値より大きく (または小さく) することで
ある範囲内に入るような、X の値を得たい
を、決定木やランダムフォレストでは「原理的に」達成できない
という点からいえば、水素化物Tc計算でランダムフォレストを使うのは間違っている
kNNも同様なのでは？


y-randomizationで過学習(オーバーフィッティング), Chance Correlation(偶然の相関)の危険度を評価！
https://datachemeng.com/y_randomization/
ref:
https://www.slideshare.net/itakigawa/ss-69269618
これが気になったので、以下で検証



[1b] 水素化物Tc計算
[1b1] 考察
昨日のDCV, y-randomizationごとのまとめ
RR
DCV:RMSE, MAE, R^2 = 34.336, 23.335, 0.645 (ave)
DCV:RMSE, MAE, R^2 =  0.879,  0.581, 0.018 (std)
EN 
DCV:RMSE, MAE, R^2 = 35.602, 24.216, 0.618 (ave)
DCV:RMSE, MAE, R^2 =  0.711,  0.295, 0.015 (std)
LASSO
DCV:RMSE, MAE, R^2 = 35.520, 24.401, 0.620 (ave)
DCV:RMSE, MAE, R^2 =  0.917,  0.687, 0.020 (std)
kNN
DCV:RMSE, MAE, R^2 = 25.630, 14.964, 0.801 (ave)
DCV:RMSE, MAE, R^2 =  1.975,  0.927, 0.031 (std)
RF
DCV:RMSE, MAE, R^2 = 23.488, 14.482, 0.833 (ave)
DCV:RMSE, MAE, R^2 =  1.502,  0.733, 0.021 (std)
GB
DCV:RMSE, MAE, R^2 = 22.881, 13.954, 0.841 (ave)
DCV:RMSE, MAE, R^2 =  2.238,  1.108, 0.031 (std)
SVR
DCV:RMSE, MAE, R^2 = 25.930, 16.193, 0.796 (ave)
DCV:RMSE, MAE, R^2 =  2.217,  0.960, 0.035 (std)
PLS
DCV:RMSE, MAE, R^2 = 38.003, 25.658, 0.564 (ave)
DCV:RMSE, MAE, R^2 =  2.020,  0.934, 0.048 (std)
GPR
DCV:RMSE, MAE, R^2 = 58.018, 31.166,-0.061 (ave)
DCV:RMSE, MAE, R^2 = 12.489,  1.696, 0.551 (std)

RR
rnd:RMSE, MAE, R^2 = 55.151, 40.756, 0.084 (ave)
rnd:RMSE, MAE, R^2 =  0.551,  0.593, 0.018 (std)
EN
rnd:RMSE, MAE, R^2 = 56.492, 41.710, 0.038 (ave)
rnd:RMSE, MAE, R^2 =  0.487,  0.290, 0.017 (std)
LASSO
rnd:RMSE, MAE, R^2 = 56.759, 41.902, 0.029 (ave)
rnd:RMSE, MAE, R^2 =  0.295,  0.226, 0.010 (std)
kNN
rnd:RMSE, MAE, R^2 = 51.154, 37.805, 0.211 (ave)
rnd:RMSE, MAE, R^2 =  1.039,  0.869, 0.032 (std)
RF
rnd:RMSE, MAE, R^2 = 40.772, 28.393, 0.498 (ave)
rnd:RMSE, MAE, R^2 =  1.632,  0.879, 0.040 (std)
GB
rnd:RMSE, MAE, R^2 = 54.249, 40.128, 0.113 (ave)
rnd:RMSE, MAE, R^2 =  0.553,  0.432, 0.018 (std)
SVR
rnd:RMSE, MAE, R^2 = 51.648, 29.654, 0.195 (ave)
rnd:RMSE, MAE, R^2 =  2.358,  2.677, 0.074 (std)
PLS
rnd:RMSE, MAE, R^2 = 57.200, 42.156, 0.014 (ave)
rnd:RMSE, MAE, R^2 =  0.341,  0.223, 0.012 (std)
GPR
rnd:RMSE, MAE, R^2 = 37.094, 22.776, 0.584 (ave)
rnd:RMSE, MAE, R^2 =  2.098,  1.091, 0.047 (std)

ランダムフォレストで高いのはともかく、他のkNNやSVRでも0に近いとは言いづらい値
一方で線形モデルは軒並み低い　→　偶然の相関がない
ノンパラメトリック回帰の場合、Xが近いなら取りあえず近いyの値が出るので、
ランダム化しても多少再現してしまうのが、0に近づけない理由？
だとしたら、ある程度は仕方ないのかもしれない


[1b2] ランダム化について検証
ramdom.py
完全にランダムなx(1-dim),y(1-dim)の100個の組について、各手法で機械学習した場合の結果
kNN: RMSE, MAE, RMSE/MAE, R^2 = 1.368, 1.178, 1.161, 0.150
RF : RMSE, MAE, RMSE/MAE, R^2 = 0.695, 0.554, 1.256, 0.750
SVR: RMSE, MAE, RMSE/MAE, R^2 = 0.902, 0.839, 1.075, 0.605
GPR: RMSE, MAE, RMSE/MAE, R^2 = 0.000, 0.000, 2.559, 1.000
GB : RMSE, MAE, RMSE/MAE, R^2 = 1.225, 1.059, 1.156, 0.180
RR : RMSE, MAE, RMSE/MAE, R^2 = 1.457, 1.258, 1.158, 0.004

100 -> 1000個に増やした場合
kNN: RMSE, MAE, RMSE/MAE, R^2 = 1.333, 1.144, 1.165, 0.102
RF : RMSE, MAE, RMSE/MAE, R^2 = 0.727, 0.554, 1.312, 0.749
SVR: RMSE, MAE, RMSE/MAE, R^2 = 1.455, 1.259, 1.155, 0.011
GPR: RMSE, MAE, RMSE/MAE, R^2 = 0.689, 0.434, 1.587, 0.772
GB : RMSE, MAE, RMSE/MAE, R^2 = 1.438, 1.259, 1.142, 0.037
RR : RMSE, MAE, RMSE/MAE, R^2 = 1.488, 1.297, 1.147, 0.001

どうやら手法とサンプルの数によっては最初から完全にランダムでもそこそこ学習できてしまうらしい
複雑なモデルでも再現できてしまう、表現力の高さが原因？
そうなると、y-randomizationで検証しても余り意味がない？
というよりは、y-randomizationでは、偶然の相関を検出できないというべきか？

サンプルの数が少ないと、偶然の相関が起きてしまうことは、金子研究室のページでも指摘されている
https://datachemeng.com/y_randomization/





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
fingerprint functionでも同様ではあるが

!?fingerprint functionとほぼ一緒だったりしないのか？
3次元と1次元とで違うのは違うが...

↓読む限りでは、ほぼ一緒というほどではなさそう


[1c2] 論文読み
J. Behler, J. Chem. Phys. 145, 170901 (2016).
"Perspective: Machine learning potentials for atomistic simulations"
https://doi.org/10.1063/1.4966192
結晶構造に対するエネルギーのモデルを機械学習で構築する方法についてのレビュー

I. INTRODUCTION
Based on these requirements, a definition of ML potential can be given. 
A ML potential
    * employs a ML method to construct a direct functional relation
     between the atomic configuration and its energy;
    * does not contain any physical approximations apart from the chosen
     reference electronic structure method used in its construction;
    * is developed using a consistent set of electronic structure data.

II. STRUCTURAL DESCRIPTION
    A. The role of the descriptor
    In Secs. II B 1–II B 4
     several descriptors which have been developed for ML potentials are discussed. 
    B. Descriptors for machine learning potentials
    1. Atom centered symmetry functions
    2. Bispectrum of the neighbor density
    3. Smooth overlap of atomic positions
    4. Coulomb matrix
# 構造の記述の仕方

III. A BRIEF SURVEY OF ML POTENTIALS
    A. Overview
    B. Neural networks
    C. Gaussian approximation potentials and kernel methods
    D. Support vector machines
    E. Spectral neighbor analysis potential
# どのようなポテンシャルを採用するか




