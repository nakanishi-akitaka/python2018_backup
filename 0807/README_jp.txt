# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 14:21:37 2018

@author: Akitaka
"""

[1c1] DCVでもR^2_DCVや正解率_DCVを実装する
ref:0731
example:
test0_kNN_AD_DCV_rgr.py
test0_kNN_AD_DCV_clf.py
library.py

各inner loopでR^2や正解率を計算、outer loopが終わったら、それらの平均・標準偏差を計算する。
→
各inner loopでyの予測値を計算、outer loopが終わったら、それらとyからR^2や正解率を計算する。
こちらの方がいい？
R^2や正解率以外も計算したときに、一々標準偏差が出ない方がスッキリする
ただし、回帰と分類とで分ける必要性はでてきた
dcv -> dcv_rgr, dcv_clf

DCVの繰り返し回数を指定して、R^2や正解率を毎回表示するではなく、
イテレーションに対しての平均値や標準偏差を出すのに変更
ループ回数が100や200とかになると、毎回ループで表示するのも分かりにくい。

ref:
https://note.nkmk.me/python-numpy-ndarray-sum-mean-axis/
https://note.nkmk.me/python-numpy-insert/
https://note.nkmk.me/python-numpy-ndarray-slice/
https://deepage.net/features/numpy-zeros.html




[1d1] 06/15 (Anaconda使い始め)からの総括
summary(ディレクトリ)に格納




[1e1] RFをtest0に
example:
test0_RF_DCV_clf.py
test0_RF_DCV_rgr.py
とりあえずは作成
特徴量の選択率を0.1~0.9からCVで決定
敢えて、特徴量の数=10, 意味のある特徴量の数=5と設定してサンプルを作成した。
