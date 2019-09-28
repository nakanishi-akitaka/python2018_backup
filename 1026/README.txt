# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 10:37:24 2018

@author: Akitaka
"""

[1b] サイトで勉強2 note.nkmk.me
Python関連記事まとめ - note.nkmk.me
https://note.nkmk.me/python-post-summary/
文字列
基礎

    文字列生成（引用符、strコンストラクタ）
    エスケープシーケンスを無視（無効化）するraw文字列
    文字列の長さ（文字数）を取得
    文字列を置換（replace, translate, re.sub, re.subn）
    文字列を分割（区切り文字、改行、正規表現、文字数）
    文字列を連結・結合（+演算子、joinなど）
    改行を含む文字列の出力、連結、分割、削除、置換
    長い文字列を複数行に分けて書く
    大文字・小文字を操作する文字列メソッド一覧

以上のページのサンプルプログラムを写経(コピペ)完了


[1c] 書籍(の英語サイト)で勉強 Python Data Science Handbook
https://jakevdp.github.io/PythonDataScienceHandbook/

Visualization with Matplotlib
https://jakevdp.github.io/PythonDataScienceHandbook/04.00-introduction-to-matplotlib.html

Simple Line Plots
https://jakevdp.github.io/PythonDataScienceHandbook/04.01-simple-line-plots.html

Simple Scatter Plots
https://jakevdp.github.io/PythonDataScienceHandbook/04.02-simple-scatter-plots.html

Visualizing Errors
https://jakevdp.github.io/PythonDataScienceHandbook/04.03-errorbars.html

Density and Contour Plots
https://jakevdp.github.io/PythonDataScienceHandbook/04.04-density-and-contour-plots.html

Histograms, Binnings, and Density
https://jakevdp.github.io/PythonDataScienceHandbook/04.05-histograms-and-binnings.html

Customizing Plot Legends
https://jakevdp.github.io/PythonDataScienceHandbook/04.06-customizing-legends.html

Customizing Colorbars
https://jakevdp.github.io/PythonDataScienceHandbook/04.07-customizing-colorbars.html

Multiple Subplots
https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html

Text and Annotation
https://jakevdp.github.io/PythonDataScienceHandbook/04.09-text-and-annotation.html

Customizing Ticks
https://jakevdp.github.io/PythonDataScienceHandbook/04.10-customizing-ticks.html

Customizing Matplotlib: Configurations and Stylesheets
https://jakevdp.github.io/PythonDataScienceHandbook/04.11-settings-and-stylesheets.html

Three-Dimensional Plotting in Matplotlib
https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html

Geographic Data with Basemap
https://jakevdp.github.io/PythonDataScienceHandbook/04.13-geographic-data-with-basemap.html

Visualization with Seaborn
https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html

Further Resources
https://jakevdp.github.io/PythonDataScienceHandbook/04.15-further-resources.html
リンクのみ


以上のページのサンプルプログラムを写経(コピペ)完了
色々なグラフの作成方法を一通り紹介している
必要な時にググればいいだけかもしれない

Pythonでデータサイエンス
https://pythondatascience.plavox.info/
ここで紹介してたレベルでも十分っぽい



[1e] 水素化物の超伝導予測
[1e1] ファイル統一
[todo] -> [done]
モデルごとにファイルを分けるとかさばるのでまとめることはできないか？
0711test5_7clf_cdv.pyで、名前とモデルとパラメータをまとめて辞書にしているので応用できる
ファイル名の設定に使えそうなページ
https://note.nkmk.me/python-string-concat/

C:\Users\Akitaka\Downloads\python\1026\Tc_3model_AD_DCV.py
現状、EN, RR, LASSOの３つのみ


[1e2] 最適化するべきパラメーターの設定ミス
0.1 - 1.0のはずが、0.1のみになっていた
早速うえのを使ってやり直し

Ridge Regression
Best parameters set found on development set:
{'model__alpha': 1.0}
C:  RMSE, MAE, R^2 = 33.894, 25.541, 0.257
CV: RMSE, MAE, R^2 = 34.458, 25.933, 0.233
P:  RMSE, MAE, R^2 = 42.513, 35.734, 0.000

Double Cross Validation
In 10 iterations, average +/- standard deviation
RMSE DCV: 34.470 (+/-0.307)
MAE  DCV: 25.939 (+/-0.215)
R^2  DCV: 0.232 (+/-0.014)

y-randomization
In 10 iterations, average +/- standard deviation
RMSE: 39.067 (+/-0.125)
MAE: 30.873 (+/-0.097)
R^2: 0.013 (+/-0.006)
32.24 seconds 

LASSO           
Best parameters set found on development set:
{'model__alpha': 0.6000000000000001}
C:  RMSE, MAE, R^2 = 33.918, 25.422, 0.256
CV: RMSE, MAE, R^2 = 34.359, 25.741, 0.237
P:  RMSE, MAE, R^2 = 39.666, 33.257, 0.000

Double Cross Validation
In 10 iterations, average +/- standard deviation
RMSE DCV: 34.662 (+/-0.317)
MAE  DCV: 25.966 (+/-0.239)
R^2  DCV: 0.223 (+/-0.014)

y-randomization
In 10 iterations, average +/- standard deviation
RMSE: 39.062 (+/-0.173)
MAE: 30.923 (+/-0.092)
R^2: 0.014 (+/-0.009)
38.04 seconds 


Elastic Net     
Best parameters set found on development set:
{'model__alpha': 0.1, 'model__l1_ratio': 0.5}
C:  RMSE, MAE, R^2 = 33.913, 25.446, 0.257
CV: RMSE, MAE, R^2 = 34.304, 25.680, 0.239
P:  RMSE, MAE, R^2 = 39.919, 33.522, 0.000

Double Cross Validation
In 10 iterations, average +/- standard deviation
RMSE DCV: 34.514 (+/-0.293)
MAE  DCV: 25.776 (+/-0.178)
R^2  DCV: 0.230 (+/-0.013)

y-randomization
In 10 iterations, average +/- standard deviation
RMSE: 39.140 (+/-0.078)
MAE: 30.843 (+/-0.117)
R^2: 0.010 (+/-0.004)
96.09 seconds 


予測されたTc top10は、３つとも92-94K ぐらい
再現率はやはり悪い
