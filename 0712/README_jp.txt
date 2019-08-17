# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 16:00:29 2018

@author: Akitaka
"""

[1f] 論文追試？
https://arxiv.org/pdf/1803.10260.pdf
ref:07/05,06,10,11

[1f1] 少しだけアップデート
追加機能：
ハイパーパラメータをグリッドサーチした上で分類＆回帰
図を横に並べて表示

サーチなし
train data: RMSE, MAE, RMSE/MAE, R^2 = 0.315, 0.164, 1.924, 0.960
test  data: RMSE, MAE, RMSE/MAE, R^2 = 0.649, 0.300, 2.160, 0.000

サーチあり
{'max_depth': 15}
train data: RMSE, MAE, RMSE/MAE, R^2 = 0.324, 0.170, 1.901, 0.957
test  data: RMSE, MAE, RMSE/MAE, R^2 = 0.622, 0.300, 2.076, 0.000

example:
test1_Eg_RFC_RFR




[1g] ダブルクロスバリデーション/二重交差検証/DCV + 21種類のクラス分類方法テスト(総当たり)
ref:0706, 0707,07/11
[1g1] 復習
example:
test2_7clf_dcv.py
