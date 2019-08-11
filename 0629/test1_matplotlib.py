# -*- coding: utf-8 -*-
"""
matplotlib で散布図 (Scatter plot) を描く
https://pythondatascience.plavox.info/matplotlib/散布図
本ページでは、Python のグラフ作成パッケージ Matplotlib を用いて散布図 (Scatter plot) 
を描く方法について紹介します。

Created on Fri Jun 29 11:57:44 2018

@author: Akitaka

matplotlib.pyplot.scatter の概要
matplotlib には、散布図を描画するメソッドとして、matplotlib.pyplot.scatter が用意されてます。

matplotlib.pyplot.scatter の使い方

matplotlib.pyplot.scatter(x, y, s=20, c=None, marker='o', cmap=None, norm=None,
                          vmin=None, vmax=None, alpha=None, linewidths=None,
                          verts=None, edgecolors=None, hold=None, data=None,
                          **kwargs)

matplotlib.pyplot.scatter の主要な引数
x, y	グラフに出力するデータ
s	サイズ (デフォルト値: 20)
c	色、または、連続した色の値
marker	マーカーの形 (デフォルト値: ‘o’= 円)
cmap	カラーマップ。c が float 型の場合のみ利用可能です。
norm	c を float 型の配列を指定した場合のみ有効。正規化を行う場合の Normalize インスタンスを指定。
vmin, vmax	正規化時の最大、最小値。 指定しない場合、データの最大・最小値となります。norm にインスタンスを指定した場合、vmin, vmax の指定は無視されます。
alpha	透明度。0(透明)～1(不透明)の間の数値を指定。
linewidths	線の太さ。
edgecolors	線の色。


"""


# グラフの出力例
# 以下例では、100 個 × 2 軸の乱数を2次元座標上にプロットします。

import numpy as np
import matplotlib.pyplot as plt
 
# 乱数を生成
x = np.random.rand(100)
y = np.random.rand(100)
 
# 散布図を描画
plt.scatter(x, y)

#%%
# サイズ、色、不透明度、線のサイズ、色を指定
plt.scatter(x, y, s=600, c="pink", alpha=0.5, linewidths="2",
            edgecolors="red")

#%%
# マーカーを指定 (星印)
plt.scatter(x, y, s=600, c="yellow", marker="*", alpha=0.5, linewidths="2",
            edgecolors="orange")

#%%
# グラフのタイトル、X 軸、Y 軸の名前 (ラベル)、グリッド線を表示
plt.scatter(x, y)
plt.title("This is a title")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.grid(True)

#%%
# カラーマップを指定して、値に応じてマーカーを着色
# 引数 s の値の大小に応じて、色の濃淡やグラデーションで表現することができます。
# 乱数を 100 件生成
value = np.random.rand(100)
 
# 散布図を表示
plt.scatter(x, y, s=100, c=value, cmap='Blues')
 
# カラーバーを表示
plt.colorbar()

#%%
# 上記に加えて正規化における最大値 (0.6)、最小値 (0.4) を指定
# (右側の凡例の目盛が変わっているのがわかるかと思います)
plt.scatter(x, y, s=100, c=value, cmap='Blues', vmin=0.4, vmax=0.6)
plt.colorbar()
