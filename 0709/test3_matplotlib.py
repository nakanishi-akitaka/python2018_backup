# -*- coding: utf-8 -*-
"""
【python】Matplotlibでグラフ作成すればデータサイエンティストの仲間入り
ref:
http://aiweeklynews.com/archives/50613195.html

Created on Mon Jul  9 15:47:41 2018

@author: Akitaka
"""

#各種ライブラリをインポート
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 平均 50, 標準偏差 15 の正規乱数を500件生成
x = np.random.normal(50, 15, 500)
 
# ヒストグラムを出力
plt.hist(x)
plt.show()

#%%
#３つのデータを作成
df1 = [36,62,78,50,65,40]
df2 = [20,55,58,70,75,60]
df3 = [50,55,61,70,65,55]
data = (df1,df2,df3)

fig = plt.figure()
ax = fig.add_subplot(111)

# 箱ひげ図をつくる
bp = ax.boxplot(data)

plt.grid()
plt.ylim([0,100])
plt.show()

#%%
num = [1, 2, 3, 4, 5, 6]
df1 = [36,62,78,50,65,40]
plt.bar(num, df1)
plt.show()

#%%
#-20から20までを100ステップに区切った列を配列として作成
x = np.linspace(-20, 20, 100)
#サイン関数
y = np.sin(x)

#plot関数は、一方の配列に対して他方の配列をプロットする。
plt.plot(x, y, marker="x")
plt.show()

#%%
# 散布図を描画
df1 = [36,62,78,50,65,40]
df2 = [20,55,58,70,75,60]
plt.scatter(df1, df2)
plt.show()
