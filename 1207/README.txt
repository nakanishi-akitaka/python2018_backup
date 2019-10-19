# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 11:35:24 2018

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


[1a2] twitterから論文読み
https://twitter.com/hirokaneko226/status/1070854765599281152
DrugBankデータベースやAlan Woodデータベースから抽出した、
リンカー・環・サイドチェインといった医薬品・農薬のフラグメントのデータベースに関する論文。
構造検査やエネルギー最適化も実施済み。
こちら 
http://chemyang.ccnu.edu.cn/ccb/database/PADFrag/
から利用可能

"PADFrag: A Database Built for the Exploration of Bioactive Fragment Space for Drug Discovery"
J. Chem. Inf. Model. 2018,  58, 9, 1725-1730
https://doi.org/10.1021/acs.jcim.8b00285
どうやって情報をあつめてデータベースにしたのかはよく分からなかった



[1b] sklearnのチートシート
【目次】Python scikit-learnの機械学習アルゴリズムチートシートを全実装・解説
http://neuro-educator.com/mlcontentstalbe/
ENとLASSOは回帰係数が0になりやすい　→　一部の特徴量が重要な場合に使用？

SGD（クラス分類）【Pythonとscikit-learnで機械学習：第1回】
http://neuro-educator.com/mlearn1/

カーネル近似（クラス分類）【Pythonとscikit-learnで機械学習：第2回】
http://neuro-educator.com/ml2/
    カーネルトリックの説明が分かりやすい

Linear SVC（クラス分類 ）（SVM Classification）【Pythonとscikit-learnで機械学習：第3回】
http://neuro-educator.com/ml3/

K近傍法（クラス分類 ）（KNeighbors Classifier）【Pythonとscikit-learnで機械学習：第4回】
http://neuro-educator.com/ml4/
    アルゴリズムは非常に単純ですが、割と精度が出る面白い手法です。
    解析的なことは苦手だが、ただ分類するだけなら得意

Kernel SVC（クラス分類）（SVM Classification）【Pythonとscikit-learnで機械学習：第5回】
http://neuro-educator.com/ml5/

ランダムフォレスト（クラス分類）Ensemble Classification【Pythonとscikit-learnで機械学習：第6回】
http://neuro-educator.com/ml6/

ナイーブベイズで自然言語処理（クラス分類）【Pythonとscikit-learnで機械学習：第7回】
http://neuro-educator.com/ml7/

ハイパーパラメータの最適化と結果の見方【Pythonとscikit-learnで機械学習：第8回】
http://neuro-educator.com/ml8/

KMeans、MiniBatch-Kmeans（クラスタ分析）【Pythonとscikit-learnで機械学習：第9回】
http://neuro-educator.com/ml9/

スペクトラルクラスタリング（SpectralClustering）（クラスタ分析）【Pythonとscikit-learnで機械学習：第10回】
http://neuro-educator.com/ml10/
    スペクトラルクラスタリングの説明は難しい。グラフ理論が関係するらしい。
    https://www.slideshare.net/pecorarista/ss-51761860

GMM・クラスタリング（クラスタ分析）【Pythonとscikit-learnで機械学習：第11回】
http://neuro-educator.com/ml11/

MeanShift（クラスタ分析）【Pythonとscikit-learnで機械学習：第12回】
http://neuro-educator.com/ml12/

VBGMM（クラスタ分析）【Pythonとscikit-learnで機械学習：第13回】
http://neuro-educator.com/ml13/
    VBGMMを理解するのは難しいらしい

SDG Regressor（回帰分析）【Pythonとscikit-learnで機械学習：第14回】
http://neuro-educator.com/ml14/

Lasso Regressor（回帰分析）【Pythonとscikit-learnで機械学習：第15回】
http://neuro-educator.com/ml15/

Ridge Regressor（回帰分析）【Pythonとscikit-learnで機械学習：第16回】
http://neuro-educator.com/ml16/

ElaticNet Regressor（回帰分析）【Pythonとscikit-learnで機械学習：第17回】
http://neuro-educator.com/ml17/

SVR Regressor Linear（回帰分析）【Pythonとscikit-learnで機械学習：第18回】
http://neuro-educator.com/ml18/

SVR Regressor rbf（回帰分析）【Pythonとscikit-learnで機械学習：第19回】
http://neuro-educator.com/ml19/

Ensemble regressor（回帰分析）【Pythonとscikit-learnで機械学習：第20回】
http://neuro-educator.com/ml20/

PCA 主成分分析（次元圧縮）【Pythonとscikit-learnで機械学習：第21回】
http://neuro-educator.com/ml21/

Kernel PCA （次元圧縮）【Pythonとscikit-learnで機械学習：第22回】
http://neuro-educator.com/ml22/

SpectralEmbedding （次元圧縮）【Pythonとscikit-learnで機械学習：第23回】
http://neuro-educator.com/ml23/

Isomap （次元圧縮）【Pythonとscikit-learnで機械学習：第24回】
http://neuro-educator.com/ml24/

LocallyLinearEmbedding （次元圧縮）【Pythonとscikit-learnで機械学習：第25回】
http://neuro-educator.com/ml25/    



[1c] モジュール作成
[todo] LWPLS, GMR モジュール化 + Tc予測
Locally-Weighted Partial Least Squares (LWPLS, 局所PLS)
 ～あのPLSが非線形性に対応！～ [Python・MATLABコードあり] 
https://datachemeng.com/locallyweightedpartialleastsquares/

Locally-Weighted Partial Least Squares (LWPLS) 
https://github.com/hkaneko1985/lwpls

https://github.com/hkaneko1985/lwpls/blob/master/Python/lwpls.py
これをベースにモジュール作成完了

https://github.com/hkaneko1985/lwpls/blob/master/Python/demo_of_lwpls.py
これをベースにしたデモ計算完了

https://github.com/hkaneko1985/lwpls/blob/master/Python/demo_lwpls_grid_search_cv.py
これをベースにした、グリッドサーチデモ計算完了
※scikit-learnのGridSearchCVは使っていない
？グリッドサーチでは、foldに分ける前に、全体でスケーリングしてから、分割してる。
　foldからtrain, testに分けたあとで、trainに合わせてスケーリングしているのではない。
　これでいい？

次に、
http://yamaguchiyuto.hatenablog.com/entry/python-advent-calendar-2014
で行われたデモ計算をやろうとしたら、エラーが出た！
上記のlwplsをそのまま使うと、yのサイズが、[n_samples, n_compnents]になってしまうため
    成分毎の計算結果を累積してyを計算する方式
    最終的には、全部累積したものだけを使うのだが、なぜか各成分までのyの計算値を逐一残している
→
最終結果のみを出力。つまり、yのサイズを[n_samples]にしたところ、
demo_lwpls_grid_search_cv.pyの方がうまくいかなくなった
→
放置することに決定
scikit-learnのGridSearchCVを使っていないプログラムに時間を使う必要性が薄い
GridSearchCVで使うことができれば十分だと判断。
→
http://yamaguchiyuto.hatenablog.com/entry/python-advent-calendar-2014
で行われたデモ計算grid_search.py, cross_validation.pyをLWPLSで実行できた！

cross_validation
Ridge:-0.05524362169074652
kNN  :-0.2369990974464181
LWPLS:-860.2157384221276

GridSearchCV
Ridge:-16.179344242620736
kNN  :-16.24411041927126
LWPLS:-1018.4442456904959

※scoring='neg_mean_squared_error'なので、小さいほど優秀
とはいっても、値が異常すぎる。そもそもグラフの見た目だけなら、RidgeもLWPLSも大差ない
恐らくは、標準化の有無の問題？
→
試しに標準化を入れてみたが、スコアに大差ない

n_components=4 -> 2にしたら、
cross_validation
LWPLS:-7.6300071396622755

GridSearchCV
LWPLS:-22.797303707648148
となり、RidgeやkNNに近い値になった。
成分数を、元のXの成分数そのままにしていたため、オーバーフィッティングが起きた？

ref
20181206ノート

scikit-learn準拠で Label propagation とか実装した
http://yamaguchiyuto.hatenablog.com/entry/2016/09/22/014202

Python: k 近傍法を実装してみる
https://blog.amedama.jp/entry/2017/03/18/140238

scikit-learn準拠の学習器を作ってgrid searchとかcross validationする
http://yamaguchiyuto.hatenablog.com/entry/python-advent-calendar-2014
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
        
sklearn準拠モデルの作り方
https://qiita.com/roronya/items/fdf35d4f69ea62e1dd91
1. クラス設計
やることは3つです。
    1.BaseEstimatorを継承
    2.回帰ならRegressorMixin、分類ならClassifierMixinを継承
    3.fit()とpredict()を実装

sklearn.utils.estimator_checksにcheck_estimator()という関数があり、
これを使うとクラス設計がsklearnのルールに従っているかチェックすることが出来ます。
2. 命名規則とかあるの？
    * fit() の後に確定する変数は変数名にサフィックスとして_を付ける 
    * 変数の最後が_の変数は__init__()で束縛しないというルールがある 
    つまりfit()した後に値が確定する変数はコンストラクタでは束縛せず、
    fit()の中で変数名にサフィックスとして_を付けて宣言します。
    例えば線形モデルなら
        係数: coef_
        バイアス項: intercept_
3. fit()する前のpredict()の挙動はどうすればいいの？
    sklearn.utils.validation.check_is_fitted()というfit()しているか否かを確かめる関数を使います。
    この関数はfit()されていなければsklearn.exceptions.NotFittedErrorを返します。
    check_is_fitted()にはselfとfit()したあとに値が確定する変数名を渡します。

[todo] モジュール化
https://github.com/hkaneko1985/kennardstonealgorithm

    FastSearchCVとして、sklearn-likeにつくる？

https://github.com/hkaneko1985/k3nerror

https://github.com/hkaneko1985/gapls_gasvr

https://datachemeng.com/gaussianmixtureregression/
https://github.com/hkaneko1985/sgmm
sklearnにGMRはない。GMMはある。
このデモプログラムは、GMMを使って、GMRを計算している


SVMの高速ハイパーパラメータ最適化を実装？　＋　DCV　＋　OCSVM
http://univprof.com/archives/16-07-14-4701508.html
https://datachemeng.com/fastoptsvrhyperparams/
