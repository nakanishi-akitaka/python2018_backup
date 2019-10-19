# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 22:54:50 2018

@author: Akitaka
"""

[1] 機械学習
[1a] サイトで勉強１　金子研究室
24周目: 12/03~12/09
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


[1a2] Adaboost
Adaboost (Adaptive Boosting) によるアンサンブル学習のやり方を解説します 
https://datachemeng.com/adaboost/
1.モデル１作成
2.モデル１で一番外したサンプルAをより正確に再現するモデル２作成
3.モデル１＋２で一番外したサンプルBをより正確に再現するモデル３作成
を、最初に決めたモデルの数になるまで繰り返す。

sklearnより
AdaBoostRegressor
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
predict(X):普通に予測
staged_predict(X)：各モデル作成段階での予測を「繰り返し」出力する
y = [n_samples, n_estimators]を返すのではなく、
y = [n_samples]を n_estimators回、返す

コードより
https://github.com/scikit-learn/scikit-learn/blob/55bf5d9/sklearn/ensemble/weight_boosting.py#L1098
    for i, _ in enumerate(self.estimators_, 1):
        yield self._get_median_predict(X, limit=i)

使用例
https://scikit-learn.org/0.18/auto_examples/ensemble/plot_adaboost_hastie_10_2.html
n_estimatorsが増えるごとに、errorが減っていく


[1a3] 論文よみ
https://twitter.com/hirokaneko226/status/1069442623121764352
温度・圧力が与えられたときに、分子の熱力学的物性を推定する論文。
アルカンの液体密度・気化熱・熱容量・気液平衡曲線・臨界温度・臨界密度・表面張力が対象。
温度・圧力条件下において化学構造のQM計算・MD計算を行い、
その結果と物性との間でニューラルネットワークを構築

"Predicting Thermodynamic Properties of Alkanes
 by High-Throughput Force Field Simulation and Machine Learning"
Zheng Gong, Yanze Wu, Liang Wu, and Huai Sun
J. Chem. Inf. Model.  XXXX, XXX, XXX-XXX
https://doi.org/10.1021/acs.jcim.8b00407

# 高圧力下での超伝導転移温度の推定に応用できる？
  何を特徴量にするか？原子の物性は、圧力下のものを使っているのか？

アブストラクトの分量的には、QM,MDがメインで、オマケ・発展としてニューラルネットワーク

[todo]->[done] MLの応用例として、これらはチェックしておく
> ML has been applied to predict crystal structures,(26−28)
> compound properties,(29) condensed matter behavior,(30)
> and for drug discovery(31) and materials design.(32) 


26
Seko, A.; Maekawa, T.; Tsuda, K.; Tanaka, I.
"Machine Learning with Systematic Density-Functional Theory Calculations:
 Application to Melting Temperatures of Single- and Binary-Component Solids."
Phys. Rev. B: Condens. Matter Mater. Phys. 2014, 89, 054303, 
https://doi.org/10.1103/PhysRevB.89.054303

説明変数
    (1) = Table IIIのみ
    (2) = Table II + III
    Table II(体積、最近接距離、凝集エネルギー、弾性定数)
    Table III(成分、原子番号、原子量、価電子数、族、周期、などの和と積)
計算方法：OLS,PLS,SVR,GPR
目的変数：融点
？　上の論文では、構造を予測しているかのように引用しているが、実際に予測しているのは融点。

27
Seko, A.; Hayashi, H.; Nakayama, K.; Takahashi, A.; Tanaka, I. 
"Representation of Compounds for Machine-Learning Prediction of Physical Properties." 
Phys. Rev. B: Condens. Matter Mater. Phys. 2017, 95, 144110, 
https://doi.org/10.1103/PhysRevB.95.144110
計算方法
    カーネルリッジ回帰
    ガウス過程回帰
    ベイズ最適化
説明変数
    Fig.1で説明。いくつかの変換を行う。
    Na^ξ(=原子数)　× Nx(=元素＋構造の情報) の行列(※)
    Nx次元の表現空間へ、Na^ξ個の点を分布させる
    分布の平均、標準偏差、歪度、尖度、共分散を求める
※Nxについて
    元素22個
    (1) atomic  number,  (2)  atomic  mass,  (3)  period  and
    (4) group  in  the  periodic  table,  (5)  first  ionization  energy,
    (6) second ionization energy, (7) electron affinity, 
    (8) Pauling electronegativity,  (9)  Allen  electronegativity,  
    (10) van  der Waals  radius,  (11)  covalent  radius,  (12)  atomic  radius,
    (13) pseudopotential radius for the s-orbital, 
    (14) pseudopotential radius  for  the p-orbital,  (15)  melting  point,
    (16) boiling point,  (17)  density,  (18)  molar  volume,  
    (19) heat  of  fusion, (20) heat of vaporization, (21) thermal conductivity,
    and (22) specific heat.
    構造4個
    histogram  representations  of 
    the  partial  radial  distribution function (PRDF),
    the generalized radial distribution function  (GRDF),
    the  BOP,
    and  the  angular  Fourier  series (AFS).
目的変数
    凝集エネルギー、LTC、融点
？　上の論文では、構造を予測しているかのように引用しているが、実際に予測しているのは融点など。
#
二元系、三元系などもまとめて扱っている!?
Nx=26で固定されているので、原子数が何個でも対応できる?
Fig2.によると、元素の情報だけよりも、構造の情報も入れた方が、明確かつ有意な改善


28.
Timoshenko, J.; Lu, D.; Lin, Y.; Frenkel, A. I. 
"Supervised Machine-Learning-Based Determination of Three-Dimensional Structure
 of Metallic Nanoparticles."
J. Phys. Chem. Lett. 2017, 8, 5091– 5098, 
https://doi.org/10.1021/acs.jpclett.7b02364
説明変数
    X線の吸収係数。実験データ
計算方法
    ニューラルネットワーク
目的変数
    配位数　→　構造何種類か figure.3
    クラス分類っぽい？

29.
Janet, J. P.; Chan, L.; Kulik, H. J. 
"Accelerating Chemical Discovery with Machine Learning:
 Simulated Evolution of Spin Crossover Complexes with an Artificial Neural Network."
J. Phys. Chem. Lett. 2018, 9, 1064– 1071, 
https://doi.org/10.1021/acs.jpclett.8b00170 
計算方法
    人工ニューラルネットワーク
目的変数
    自由エネルギー？
説明変数
    Figure.1 金属特性、局所/偏在配位子特性など
GA＋DFTが数日かかるのに比べて、機械学習は数秒で終わる

30.
Sun, Y. T.; Bai, H. Y.; Li, M. Z.; Wang, W. H. 
"Machine Learning Approach for Prediction and Understanding of Glass-Forming Ability."
J. Phys. Chem. Lett. 2017, 8, 3434– 3439, 
https://doi.org/10.1021/acs.jpclett.7b01046
計算方法
    SVMクラス分類
目的変数
    ガラス生成活性？glass-forming ability (GFA) 
説明変数
    11個　原子の質量(w1,w2)、混合エントロピー(dH)、原子半径(r1,r2)、液相温度(Tliq1,Tliq2)
    仮想液相温度(Tfic)、液相温度の差分(dTliq)、原子の個数？(c1,c2)

31.
Agarwal, S.; Dugar, D.; Sengupta, S. 
"Ranking Chemical Structures for Drug Discovery: A New Machine Learning Approach."
J. Chem. Inf. Model. 2010, 50, 716– 731, 
https://doi.org/10.1021/ci9003865
スクリーニングに対する化合物のランキング(優先順位付け)の需要が急増
機械学習が使われている
    SVMなどのクラス分類が化合物を活性/非活性で区別するのに使われている
    PLSやSVRなどのQSAR回帰分析が化合物の生理活性予測に使われている
最近、新しい機械学習の手法(=ランキング)が開発され、情報検索でのランキングなどに使われている
この新しい手法を化学構造のランキングに適用した
    上記の仮想スクリーニング＋クラス分類や、QSAR分析＋回帰分析よりも高精度

32.
Kim, E.; Huang, K.; Saunders, A.; McCallum, A.; Ceder, G.; Olivetti, E. 
"Materials Synthesis Insights from Scientific Literature via Text Extraction
 and Machine Learning."
Chem. Mater. 2017, 29, 9436– 9444, 
https://doi.org/10.1021/acs.chemmater.7b03500
# 私の理解
論文PDFをplain-textに変換して、物質の合成条件のデータを収集している？
そこから機械学習で、合成条件を予測させている？
Figure.3のROC見ると、あまり精度はよくない？


[1a4] ランキングについて検索
定義した重要度に基づいて情報を並べるための機械学習の一手法をランキング学習と言います。
https://aidemy.net/courses/5110

DSIRNLP#1 ランキング学習ことはじめ
https://www.slideshare.net/sleepy_yoshi/dsirnlp1

CatBoostのランク学習（Learning to rank）をためそう
https://orizuru.io/blog/machine-learning/catboost/
ランク学習とは文書や商品などのランキングを学習する方法のことをいいます。
たとえば、どこかの検索エンジンでキーワードを入力して検索をおこなうと色々なウェブページがでてきますが、
これらのウェブページをどういう順番で表示するのが良いのかを学習するのがランク学習になります。
ランク学習を利用することで、検索エンジンではユーザーに見られる確率が高いページを上位に表示したり、
ショッピングサイトではおすすめの商品を提示することができるようになります。

http://www.kamishima.net/archive/mldm-overview.pdf
ランキング学習 (learning to rank)／ 順序回帰 (ordinal regression)
：予測変数が，上中下といった順序関係のある離散値
（順序変量）である場合．情報検索で適合する文書を順位付けする応用など 

情報推薦におけるユーザの価値判断基準モデルに基づくコンテキスト依存型ランキング方式
https://ipsj.ixsq.nii.ac.jp/ej/?action=pages_view_main
&active_action=repository_view_main_item_detail&item_id=60719&item_no=1
&page_id=13&block_id=8
ランキングを提供する手法としては、Multi-Class SVM, Ranking SVMを用いた方法など

# つまり、ランキングとは以下のようなもだと考えられる？
1.クラス間に大小関係がある多クラス分類
2.不連続値に対する回帰分析


[2] todo処理
[TODO]??? SSCHAをpython使って書き直し&非調和効果計算についてSrTiO3で再計算？
→[stop] 苦労する割に成果がないので辞めておく


[todo] LWPLS, GMR モジュール化 + Tc予測

scikit-learn準拠の学習器を作ってgrid searchとかcross validationする
http://yamaguchiyuto.hatenablog.com/entry/python-advent-calendar-2014
→
サンプルプログラムをコピペ完了
※モジュールの場所移動などバージョンの違いによる部分に注意

scikit-learn準拠にするには？
やること
    sklearn.base.BaseEstimatorを継承する
    回帰ならRegressorMixinを（多重）継承する
    分類ならClassifierMixinを（多重）継承する
    fitメソッドを実装する
        学習データとラベルを受け取って学習したパラメータをフィールドにセットする
        initでパラメータをいじる操作を入れるとgrid searchが動かなくなる（後述）
    predictメソッドを実装する
        テストデータを受け取ってラベルのリストを返す
実装例（リッジ回帰）
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class RidgeRegression(BaseEstimator, RegressorMixin):
    def __init__(self,lamb=1.0):
        self.lamb = lamb

    def fit(self,X,y):
        A = np.dot(X.T,X) + self.lamb * np.identity(X.shape[1])
        b = np.dot(X.T,y)
        self.coef_ = np.linalg.solve(A,b)
        return self

    def predict(self,X):
        return np.dot(X,self.coef_)

Python: k 近傍法を実装してみる
https://blog.amedama.jp/entry/2017/03/18/140238
ここでのkNN実装で、↑の記事のように、GridSearchCVや、cross_val_scoreをできる？
→
エラー
GridSearchCV, scoreメソッドがない
cross_val_score, get_paramsメソッドがない
おそらくは、ClassifierMixinを（多重）継承していないため

sklearnのプログラムでも、
https://github.com/scikit-learn/scikit-learn/blob/55bf5d9/sklearn/neighbors/classification.py
> from ..base import ClassifierMixin
> class KNeighborsClassifier(NeighborsBase, KNeighborsMixin,
>     SupervisedIntegerMixin, ClassifierMixin):
としている
そして、このソースコードには、scoreやget_paramsメソッドがない
おそらくは、共通であるため、ClassifierMixinで定義されている
→
継承すればOK?
と思ったがダメ
そもそもpredictで返す値が1つのみであるため
LOOしかできない！

my_library.pyのad_knnのを参考にして、複数のxでも対応できるようにしないとダメだろう
->
できた！
注意点
self._k = k ではダメ。_k=Noneになってしまっている様子
self.k = k のようにアンダースコアを消す。


ref:
https://github.com/hkaneko1985/sgmm
sklearnにGMRはない。GMMはある。
このデモプログラムは、GMMを使って、GMRを計算している

[todo] モジュール化
https://github.com/hkaneko1985/kennardstonealgorithm
https://github.com/hkaneko1985/fastoptsvrhyperparams
    FastSearchCVとして、sklearn-likeにつくる？
https://github.com/hkaneko1985/k3nerror
https://github.com/hkaneko1985/gapls_gasvr
https://datachemeng.com/locallyweightedpartialleastsquares/
https://github.com/hkaneko1985/locallyweightedpartialleastsquares/
https://datachemeng.com/gaussianmixtureregression/
https://github.com/hkaneko1985/sgmm
[todo]
SVMの高速ハイパーパラメータ最適化を実装？　＋　DCV　＋　OCSVM
http://univprof.com/archives/16-07-14-4701508.html
https://datachemeng.com/fastoptsvrhyperparams/



参考にする！
sklearn準拠モデルの作り方
https://qiita.com/roronya/items/fdf35d4f69ea62e1dd91
scikit-learn準拠で Label propagation とか実装した
http://yamaguchiyuto.hatenablog.com/entry/2016/09/22/014202
scikit-learn準拠の学習器を作ってgrid searchとかcross validationする
http://yamaguchiyuto.hatenablog.com/entry/python-advent-calendar-2014
Python: k 近傍法を実装してみる
https://blog.amedama.jp/entry/2017/03/18/140238
https://www.sejuku.net/blog/25587
http://maku77.github.io/python/env/create-module.html
https://qiita.com/Tocyuki/items/fb99d9bdb71875843357
https://qiita.com/Usek/items/86edfa0835292c80fff5
