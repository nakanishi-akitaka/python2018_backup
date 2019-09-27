# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 10:38:33 2018

@author: Akitaka
"""

[1b] サイトで勉強2 note.nkmk.me
Python関連記事まとめ - note.nkmk.me
https://note.nkmk.me/python-post-summary/
タプル
    要素が1個のタプルには末尾にカンマが必要
    タプルやリストをアンパック（複数の変数に展開して代入）

辞書
    辞書を作成するdict()と波括弧、辞書内包表記
    辞書に要素を追加、辞書同士を連結（結合）
    辞書の要素を削除するclear, pop, popitem, del
    辞書のキー・値の存在を確認、取得（検索）
    辞書のgetメソッドでキーから値を取得（存在しないキーでもOK）
    辞書のforループ処理（keys, values, items）
    辞書の値からキーを抽出
    辞書のキーと値を入れ替える
    辞書の値の最大値・最小値とそのキーを取得
    複数の辞書のキーに対する集合演算（共通、和、差、対象差）
    辞書のリストから特定のキーの値のリストを取得
    辞書のリストを特定のキーの値に従ってソート
    順序付き辞書OrderedDictの使い方

以上のページのサンプルプログラムを写経完了


[1c] 書籍(の英語サイト)で勉強 Python Data Science Handbook
https://jakevdp.github.io/PythonDataScienceHandbook/

Combining Datasets: Concat and Append
https://jakevdp.github.io/PythonDataScienceHandbook/03.06-concat-and-append.html

Combining Datasets: Merge and Join
https://jakevdp.github.io/PythonDataScienceHandbook/03.07-merge-and-join.html

Aggregation and Grouping
https://jakevdp.github.io/PythonDataScienceHandbook/03.08-aggregation-and-grouping.html

Pivot Tables
https://jakevdp.github.io/PythonDataScienceHandbook/03.09-pivot-tables.html

Vectorized String Operations
https://jakevdp.github.io/PythonDataScienceHandbook/03.10-working-with-strings.html

Working with Time Series
https://jakevdp.github.io/PythonDataScienceHandbook/03.11-working-with-time-series.html

High-Performance Pandas: eval() and query()
https://jakevdp.github.io/PythonDataScienceHandbook/03.12-performance-eval-and-query.html

Further Resources
https://jakevdp.github.io/PythonDataScienceHandbook/04.15-further-resources.html
※参考リンクのみ





[1e] 水素化物の超伝導予測
リッジ回帰(Ridge Regression, RR), 
Least Absolute Shrinkage and Selection Operator (LASSO), 
Elastic Net (EN)
の３つでもやってみる
精度が悪い事は承知の上

ref:20180808
ENのパラメータについて
ややこしい事に、sklearnでは、金子研や他のサイトと、記号の使い方が異なる。
λ [ α*Sum w**2 (RR) + (1-α)*Sum |w| (LASSO) ] 金子研究室など
α [ 1/2*(1-λ)*Sum w**2 (RR) + λ*Sum |w| (LASSO)] sklearn
※λ = l1_ratio

RR
Best parameters set found on development set:
{'model__alpha': 0.1}
C:  RMSE, MAE, R^2 = 33.894, 25.547, 0.257
CV: RMSE, MAE, R^2 = 34.286, 25.975, 0.240
P:  RMSE, MAE, R^2 = 42.670, 35.866, 0.000

Predicted Tc is written in file Tc_RR_AD_DCV.csv

Double Cross Validation
In 10 iterations, average +/- standard deviation
RMSE DCV: 34.276 (+/-0.289)
MAE  DCV: 25.884 (+/-0.219)
R^2  DCV: 0.241 (+/-0.013)

y-randomization
In 10 iterations, average +/- standard deviation
RMSE: 39.184 (+/-0.117)
MAE: 30.918 (+/-0.222)
R^2: 0.008 (+/-0.006)
16.72 seconds 


LASSO
Best parameters set found on development set:
{'model__alpha': 0.1}
C:  RMSE, MAE, R^2 = 33.895, 25.525, 0.257
CV: RMSE, MAE, R^2 = 34.172, 25.780, 0.245
P:  RMSE, MAE, R^2 = 42.153, 35.425, 0.000

Predicted Tc is written in file Tc_LASSO_AD_DCV.csv

Double Cross Validation
In 10 iterations, average +/- standard deviation
RMSE DCV: 34.290 (+/-0.165)
MAE  DCV: 25.799 (+/-0.143)
R^2  DCV: 0.240 (+/-0.007)

y-randomization
In 10 iterations, average +/- standard deviation
RMSE: 39.164 (+/-0.080)
MAE: 30.897 (+/-0.179)
R^2: 0.009 (+/-0.004)
17.61 seconds 



EN
Best parameters set found on development set:
{'model__alpha': 0.1, 'model__l1_ratio': 0.1}
C:  RMSE, MAE, R^2 = 33.944, 25.396, 0.255
CV: RMSE, MAE, R^2 = 34.556, 25.841, 0.228
P:  RMSE, MAE, R^2 = 38.430, 32.235, 0.000

Predicted Tc is written in file Tc_EN_AD_DCV.csv

Double Cross Validation
In 10 iterations, average +/- standard deviation
RMSE DCV: 34.523 (+/-0.388)
MAE  DCV: 25.761 (+/-0.265)
R^2  DCV: 0.230 (+/-0.017)

y-randomization
In 10 iterations, average +/- standard deviation
RMSE: 39.156 (+/-0.165)
MAE: 30.954 (+/-0.114)
R^2: 0.009 (+/-0.008)
15.44 seconds 
