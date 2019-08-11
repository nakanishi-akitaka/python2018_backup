# -*- coding: utf-8 -*-
"""
matplotlib で折れ線グラフを描く
https://pythondatascience.plavox.info/matplotlib/折れ線グラフ
本ページでは、Python のグラフ作成パッケージ Matplotlib を用いて折れ線グラフ (line chart)
を描く方法について紹介します。

Created on Fri Jun 29 12:33:15 2018

@author: Akitaka



matplotlib.pyplot.plot の概要
matplotlib には、折れ線グラフを描画するメソッドとして、matplotlib.pyplot.plot が用意されてます。

matplotlib.pyplot.plot の使い方

matplotlib.pyplot.plot メソッド は、内部的に matplotlib.lines.Line2D クラスを参照しているため、使い方は、matplotlib.lines.Line2D クラスを参考にします。
matplotlib.lines.Line2D(xdata, ydata, linewidth=None, linestyle=None, color=None,
                        marker=None, markersize=None, markeredgewidth=None,
                        markeredgecolor=None, markerfacecolor=None,
                        markerfacecoloralt='none',fillstyle=None, antialiased=None,
                        dash_capstyle=None, solid_capstyle=None,
                        dash_joinstyle=None, solid_joinstyle=None, pickradius=5,
                        drawstyle=None, markevery=None, **kwargs)

matplotlib.pyplot.Line2D クラスの主要な引数
xdata (必須)	X 軸方向の数値
ydata (必須)	Y 軸方向の数値
linewidth	線の太さ
linestyle	線のスタイル。‘solid’ (実線), ‘dashed’ (破線), ‘dashdot’ (破線&点線), ‘dotted’ (点線) から指定。(デフォルト値:”solid”)
color	線の色
marker	マーカーの種類。参考: matplotlib.markers (デフォルト値:”None”)
markersize	マーカーの大きさ
markeredgewidth	マーカーの枠線の太さ
markeredgecolor	マーカーの枠線の色
markerfacecolor	マーカーの塗りつぶしの色
markerfacecoloralt	マーカーの塗りつぶしの色 2。fillstyle で left, right, bottom, top を指定した際、塗りつぶされない領域が ‘markerfacecoloralt’ で指定された色となります。 (デフォルト値: ‘none’)
fillstyle	マーカーの塗りつぶしのスタイル。‘full’ (全体), ‘left’ (左半分), ‘right’ (右半分), ‘bottom’ (下半分), ‘top’ (上半分), ‘none’ (塗りつぶしなし) から選択。
antialiased	アンチエイリアス (線を滑らかに描画する処理) を適用するかどうか。False または True から選択。

"""
# グラフの出力例
# 5 件のデータについて折れ線グラフを出力します。
import numpy as np
import matplotlib.pyplot as plt
 
# 折れ線グラフを出力
left = np.array([1, 2, 3, 4, 5])
height = np.array([100, 300, 200, 500, 400])
plt.plot(left, height)

#%%
# 線の太さや色など見た目の設定
# 線の太さを 4、線の色を赤 (red) に指定
plt.plot(left, height, linewidth=4, color="red")

#%%
#複数の線を出力。上から、実線、破線、破線&点線、点線。
plt.plot(left, height, linestyle="solid")
plt.plot(left, height+100, linestyle="dashed")
plt.plot(left, height+200, linestyle="dashdot")
plt.plot(left, height+300, linestyle="dotted")

#%%
# マーカーを表示
# マーカー（丸印: o）を出力
plt.plot(left, height, marker="o", markersize=10, markeredgecolor="red",
         markerfacecolor="yellow")

#%%
# マーカー（丸印: o）、枠線なしで出力
plt.plot(left, height, marker="o", markersize=10, markeredgecolor="red",
         markerfacecolor="yellow", markeredgewidth=0)

#%%
# マーカーをダイヤモンド型 (D)、サイズ 12、マーカーの枠線のサイズ 3、マーカーの枠線の色 青、マーカーの塗りつぶしの色 水色で出力。
plt.plot(left, height, marker="D", markersize=12, markeredgewidth=3,
         markeredgecolor="blue", markerfacecolor="lightblue")

#%%
# マーカーをダイヤモンド型 (s)、サイズ 12、マーカーの枠線のサイズ 3、マーカーの枠線の色 青、マーカーの塗りつぶしの色1 青、マーカーの塗りつぶしの色2 赤、マーカーの塗りつぶし方法 左半分で出力。
plt.plot(left, height, marker="s", markersize=20, markeredgewidth=2, 
         markeredgecolor="black", markerfacecolor="blue",
         markerfacecoloralt="red", fillstyle="left")

#%%
# アンチエイリアス (線を滑らかにする処理) を無効化 (上側)
plt.plot(left, height, linewidth=5, antialiased=False)
plt.plot(left, height-50, linewidth=5)

#%%
# 凡例、タイトル、ラベルなど
# グラフのタイトル、X 軸、Y 軸の名前 (ラベル)、グリッド線を表示
plt.plot(left, height)
plt.title("This is a title")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.grid(True)

#%%
# 左上 (loc=2) に凡例を追加
p1 = plt.plot(left, height, linewidth=2)
p2 = plt.plot(left, height/2, linewidth=2, linestyle="dashed")
plt.legend((p1[0], p2[0]), ("Class 1", "Class 2"), loc=2)

#%%
# 棒グラフと折れ線グラフを混在して出力
bar_height = np.array([100, 200, 300, 400, 500])
line_height = np.array([100, 300, 200, 500, 400])
 
# 棒グラフを出力
fig, ax1 = plt.subplots()
ax1.bar(left, bar_height, align="center", color="royalblue", linewidth=0)
ax1.set_ylabel('Axis for bar')
 
# 折れ線グラフを出力
ax2 = ax1.twinx()
ax2.plot(left, line_height, linewidth=4, color="crimson")
ax2.set_ylabel('Axis for line')
