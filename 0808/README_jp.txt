# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:29:27 2018

@author: Akitaka
"""


[1b3] ガウス過程回帰がどちらでもでてきたので詳しく勉強してみる
example:
0622/test2_17rgr_TMXO2_gap.py　(回帰)
0626/test7_21clf_TMXO2_gap.py　(分類)

ガウス過程による回帰(Gaussian Process Regression, GPR)
～予測値だけでなく予測値のばらつきも計算できる！～ 
https://datachemeng.com/gaussianprocessregression/
http://univprof.com/archives/16-06-24-4207217.html
推定値に加えてその推定値の標準偏差も一緒に出力します。
GPではモデル構築用データセットの目的変数の値に100%合うような回帰モデルが構築されますので、
目的変数のr2C・RMSEC を計算することや、実測値と計算値とのプロットを見ることは意味がありません。





[1c1] RFをtest0に
example:
test0_RF_DCV_clf.py
test0_RF_DCV_rgr.py
とりあえずは作成
特徴量の選択率を0.1~0.9からCVで決定
敢えて、特徴量の数=10, 意味のある特徴量の数=5と設定してサンプルを作成した。


https://datachemeng.com/randomforest/
サブデータセットの数＝決定木の数も、特徴量の割合もCVを用いて最適化する。
決定木の深さはとりあえず深くていい。
特徴量の重要性について
ここでは、OOBを用いて計算するとあるが、
sklearnでは、(ソースコードを読む限り)ジニ重要度というものを計算している

ねこし　金子研究室とは色々いってること違う
http://univprof.com/archives/16-04-06-2889192.html
https://note.mu/univprof/n/n7d9eb3ce2c74
サブデータセットの数＝決定木の数は設定する(最適化しない)が、特徴量の割合は最適化する。
ただしCVではなく、OOBを使ってHPを最適化する。
最適化した後も、R^2_CとR^2_OOBを計算している




[1c2] ADを決める
kNNやOCSVMでADを決めるのをモジュール化する
refのプログラムをミスに見て気づいた！
(1) ADのスレッショルドを決める際、多くのX_testが入るようにしていた。
    X_testではなく、X_trainでないと意味がない！
    X_train同士の、k個となりまでの距離の平均を並べたときに、
    その大半(例：99.7%)が入るような距離をスレッショルドとする。
(2) 距離が遠い方(=データ密度が少ない方)をADないに入れるようにしていた！

ver.1 sklearnのNeighberを使って距離を計算する
ver.2 scipyのcdotを使って計算
どちらも同じ値になる
→
RFにも実装。一方、kNNの信頼性は削って、統一することにした。

ref:
https://datachemeng.com/pythonassignment/
    課題15: 課題14と同様にPLSモデルを構築せよ。
    さらに、k最近傍法(k-nearest neighbor, kNN)により
    モデルの適用範囲(Applicability Domain, AD)を設定せよ。
https://datachemeng.com/wp-content/uploads/assignment15.py

example:
my_library.py
test0_kNN_AD_DCV.clf.py
test0_kNN_AD_DCV.rgr.py
test0_RF_AD_DCV.clf.py
test0_RF_AD_DCV.rgr.py
