# -*- coding: utf-8 -*-
"""
matplotlib で指定可能なマーカーの名前
https://pythondatascience.plavox.info/matplotlib/マーカーの名前
Created on Fri Jun 29 13:45:04 2018

@author: Akitaka
"""

'''
このページでは、Python のグラフ描画ライブラリの matplotlib で散布図などを出力する際に指定可能なマーカーの名前を紹介します。

matplotlib.markers クラスで定義されているマーカーの種類
“.”	point	点
“,”	pixel	四角形
“o”	circle	円
“v”	triangle_down	下向き三角形
“^”	triangle_up	上向き三角形
“<"	triangle_left	左向き三角形
“>”	triangle_right	右向き三角形
“1”	tri_down	Y 字
“2”	tri_up	Y 字 (上下反転)
“3”	tri_left	Y 字 (90 度時計回り)
“4”	tri_right	Y 字 (90 度反時計回り)
“8”	octagon	八角形
“s”	square	四角形
“p”	pentagon	五角形
“*”	star	星印
“h”	hexagon1	六角形 (縦長)
“H”	hexagon2	六角形 (横長)
“+”	plus	プラス (+) 印
“x”	x	バツ (×) 印
“D”	diamond	菱形 (ダイヤモンド)
“d”	thin_diamond	細い菱形 (ダイヤモンド)
“|”	vline	縦線
“_”	hline	横線
“None”	nothing	マーカーなし
None	nothing	マーカーなし
” “	nothing	マーカーなし
“”	nothing	マーカーなし
‘$…$’	render the string using mathtext.	Mathtext を用いて数式やギリシャ文字を指定

'''

import matplotlib.pyplot as plt
import numpy as np
 
x = np.arange(1, 11)
y1 = np.repeat(3, 10)
y2 = np.repeat(2, 10)
y3 = np.repeat(1, 10)
 
markers1 = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3"]
markers2 = ["4", "8", "s", "p", "*", "h", "H", "+", "x", "D"]
markers3 = ["d", "|", "_", "None", None, "", "$x$",
            "$\\alpha$", "$\\beta$", "$\\gamma$"]
for i in x-1:
  plt.scatter(x[i], y1[i], s=300, marker=markers1[i])
  plt.scatter(x[i], y2[i], s=300, marker=markers2[i])
  plt.scatter(x[i], y3[i], s=300, marker=markers3[i])
  
