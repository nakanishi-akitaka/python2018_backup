# -*- coding: utf-8 -*-
"""
人工知能できのこ派とたけのこ派を予測する【python決定木】
ref:
http://aiweeklynews.com/archives/50638956.html
Created on Sat Jul  7 22:40:22 2018

@author: Akitaka
"""
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import export_graphviz

# 身長、体重、たけのこ派の有無のデータを定義
# 身長、体重、たけのこ派の有無のデータを用意します（疑似データです）。
# takenoko列の0がきのこ派、1がたけのこ派です。
#データフレームの作成
df = pd.DataFrame({'height': [170, 181, 173, 159, 176, 185, 145,154, 156,170],
                   'weight': [60, 59, 61, 51, 62, 70, 50,56,60,50],
                   'takenoko': [1, 0, 0, 1, 0, 0, 1, 1, 0,1]})

#データを説明変数と目的変数に分割して格納
x = df.loc[:, ['height','weight']].values
y = df.loc[:, 'takenoko'].values

# scikit-learnのDecisionTreeClassifier（決定木）で学習
# 最大の深さを４とする決定木を定義して、fit関数で学習を行います。
# 決定木の作成
clf = tree.DecisionTreeClassifier(max_depth=4)
clf = clf.fit(x, y)

# 正解率でモデルの評価
# 決定木が予測したデータと実際のデータを比較して、正解率を測定します。
# 予測データ作成
y_pre = clf.predict(x)
print(y_pre)

# 正解率
print(accuracy_score(y, y_pre))

# 結果：精度100%のモデルが出来ました。


# 決定木の予測結果
# numpyのmeshgrid関数を活用してメッシュ状の座標を作成し、全てのメッシュに予測結果をプロットします。
# 教師データの取りうる範囲 +-1 を計算する
train_x_min = x[:, 0].min() - 1
train_y_min = x[:, 1].min() - 1
train_x_max = x[:, 0].max() + 1
train_y_max = x[:, 1].max() + 1

# 教師データの取りうる範囲でメッシュ状の座標を作る
xx, yy = np.meshgrid(
    np.arange(train_x_min, train_x_max, 0.5),
    np.arange(train_y_min, train_y_max, 0.5),
)

# メッシュの座標を学習したモデルで判定させる
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# 各点の判定結果をグラフに描画する
plt.pcolormesh(xx, yy, Z.reshape(xx.shape), cmap=ListedColormap(
        ['#FFAAAA', '#AAAAFF']))
plt.show()

# 決定木の可視化
# export_graphviz関数で、決定木のdotファイル（決定木の分岐ロジックが書いてある）を作成できます。
# 作成したファイルをwindows用のGveditツールで開くと、
# 以下のような決定木を可視化するグラフが作成できます。
# 決定木の可視化
export_graphviz(clf, out_file="tree1.dot", filled=True,rounded=True)

