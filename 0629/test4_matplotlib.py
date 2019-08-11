# -*- coding: utf-8 -*-
"""
matplotlib で円グラフを描く
https://pythondatascience.plavox.info/matplotlib/円グラフ
本ページでは、Python のグラフ作成パッケージ、Matplotlib を用いて円グラフ (pie chart)
を描く方法について紹介します。

Created on Fri Jun 29 13:05:31 2018

@author: Akitaka

matplotlib.pyplot.pie の概要
matplotlib には円グラフを描画するメソッドとして、matplotlib.pyplot.pie が用意されています。

matplotlib.pyplot.pie の使い方
matplotlib.pyplot.pie(x, explode=None, labels=None, colors=None, autopct=None,
                      pctdistance=0.6, shadow=False, labeldistance=1.1,
                      startangle=None, radius=None, counterclock=True,
                      wedgeprops=None, textprops=None, center=(0, 0),
                      frame=False, hold=None, data=None)

matplotlib.pyplot.pie の主要な引数
x (必須)	各要素の大きさを配列で指定。
explode	各要素を中心から離して目立つように表示。
labels	各要素のラベル。
colors	各要素の色を指定。
autopct	構成割合をパーセンテージで表示。 (デフォルト値: None)
pctdistance	上記のパーセンテージを出力する位置。円の中心 (0.0) から円周 (1.0) を目安に指定。autopct を指定した場合のみ有効。 (デフォルト値: 0.6)
shadow	True に設定すると影を表示。 (デフォルト値: False)
labeldistance	ラベルを表示する位置。円の中心 (0.0) から円周 (1.0) を目安に指定。 (デフォルト値: 1.1)
startangle	各要素の出力を開始する角度。 (デフォルト値: None)
radius	円の半径。 (デフォルト値: 1)
counterclock	True に設定すると時計回りで出力。False に設定すると反時計回りで出力。 (デフォルト値: True)
wedgeprops	ウェッジ (くさび形の部分) に関する指定。枠線の太さなどを設定可能。 (デフォルト値: None)
textprops	テキストに関するプロパティ。 (デフォルト値: None)

"""
# グラフの出力例
import numpy as np
import matplotlib.pyplot as plt
 
# 円グラフを描画
x = np.array([100, 200, 300, 400, 500])
plt.pie(x)

#%%
# スタイルに関するカスタマイズ
# ラベルを表示
label = ["Apple", "Banana", "Orange", "Grape", "Strawberry"]
plt.pie(x, labels=label)

#%%
# 開始角度を 90 度, 時計回りに設定
plt.pie(x, labels=label, counterclock=False, startangle=90)

#%%
# 環境によっては必要なカスタマイズ
# 真円になるようにアスペクト比を固定
# plt.axis('equal') を指定することで、Jupyter 等の環境で楕円のようなつぶれた形で
# 出力されることを防ぐことができます。
plt.pie(x, labels=label, counterclock=False, startangle=90)
plt.axis('equal')

#%%
# 背景を白に設定
# PyCharm など一部の環境では、背景がグレーで出力されるため、以下の方法で背景色を設定します。
fig = plt.figure()
fig.patch.set_facecolor('white')
plt.pie(x, labels=label, counterclock=False, startangle=90)

#%%
# 各要素のスタイルのカスタマイズ
# 1 つめの要素を 20% ずらして目立たせて表示
plt.pie(x, labels=label, counterclock=False, startangle=90,
        explode=[0.2, 0, 0, 0, 0])
plt.axis('equal')

#%%
# 各要素の色を指定
colors = ["lightpink", "yellow", "gold", "slateblue", "lightcoral"]
plt.pie(x, labels=label, counterclock=False, startangle=90, colors=colors)
plt.axis('equal')

#%%
# 各要素の色を濃淡で指定
fig = plt.figure()
fig.patch.set_facecolor('white') # 背景を白に設定
colors2 = ["0.1", "0.3", "0.5", "0.7", "0.9"]
plt.pie(x, labels=label, counterclock=False, startangle=90, colors=colors2)
plt.axis('equal')

#%%
# 影を設定
plt.pie(x, labels=label, counterclock=False, startangle=90, shadow=True)
plt.axis('equal')

#%%
# ラベルを中心から円周の間の 50% の位置に表示
plt.pie(x, labels=label, counterclock=False, startangle=90, labeldistance=0.5)
plt.axis('equal')

#%%
# 各要素のスタイルに関する設定
# 枠線の太さを 3, 線の色を白に指定
plt.pie(x, labels=label, counterclock=False, startangle=90,
        wedgeprops={'linewidth': 3, 'edgecolor':"white"})
plt.axis('equal')

#%%
# 枠線の太さを 0 (枠線なし) に指定
plt.pie(x, labels=label, counterclock=False, startangle=90,
        wedgeprops={'linewidth': 0})
plt.axis('equal')

#%%
# ラベル・テキストに関する設定
# テキストの色を白色、太字に設定
plt.pie(x, labels=label, counterclock=False, startangle=90, labeldistance=0.5,
        textprops={'color': "white", 'weight': "bold"})
plt.axis('equal')

#%%
# 構成割合をプロット。小数点以下 1 桁まで出力
plt.pie(x, labels=label, counterclock=False, startangle=90, autopct="%1.1f%%")
plt.axis('equal')

#%%
# 構成割合をプロット。小数点以下 1 桁まで、中心から円周の間の 70% の位置に出力
plt.pie(x, labels=label, counterclock=False, startangle=90, autopct="%.1f%%",
        pctdistance=0.7)
plt.axis('equal')

#%%
# より高度なパラメータの設定
# ラベルの横位置を各要素の中央に合わせる
patches, texts = plt.pie(x, labels=label, counterclock=False, startangle=90,
                         labeldistance=0.5,
                         textprops={'color': "white", 'weight': "bold"})
for t in texts:
  t.set_horizontalalignment('center')
plt.axis('equal')
  
#%%
# ラベルの横位置を各要素の中央に合わせ、サイズを 18 に指定
patches, texts = plt.pie(x, labels=label, counterclock=False, startangle=90,
                         labeldistance=0.6,
                         textprops={'color': "white", 'weight': "bold"})
for t in texts:
  t.set_horizontalalignment('center')
  t.set_size(18)
plt.axis('equal')


#%%
# ドーナツチャート (ドーナツグラフ)
# 内側に白色で円を描画することで、ドーナツグラフを作成できます。(参考)
# https://medium.com/@krishnakummar/donut-chart-with-python-matplotlib-d411033c960b#.xr7rzhq8h
plt.pie(x, labels=label, counterclock=False, startangle=90)
 
# 中心 (0,0) に 60% の大きさで円を描画
centre_circle = plt.Circle((0,0),0.6,color='black', fc='white',linewidth=1.25)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')

#%%
# 2 重ドーナツチャート (2 重ドーナツグラフ)
# 円グラフの中に円グラフを出力することで、2 重ドーナツチャートを出力できます。
# radius パラメータで、内側の円グラフの大きさを調整しています。
# 円グラフ (外側)
x1 = np.array([100, 200, 300, 400, 500])
plt.pie(x1, labels=label, counterclock=False, startangle=90)
 
# 円グラフ (内側, 半径 70% で描画)
x2 = np.array([150, 250, 300, 350, 450])
plt.pie(x2, counterclock=False, startangle=90, radius=0.7)
 
# 中心 (0,0) に 40% の大きさで円を描画
centre_circle = plt.Circle((0,0),0.4,color='black', fc='white',linewidth=1.25)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')


