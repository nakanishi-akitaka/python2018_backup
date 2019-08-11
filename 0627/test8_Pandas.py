# -*- coding: utf-8 -*-
"""
Pandas のデータフレームをソートする
このページでは、インデックスや値に基づいてデータフレームをソート(並び替え)する方法について紹介します。
Created on Wed Jun 27 13:06:57 2018

@author: Akitaka
"""
# https://pythondatascience.plavox.info/pandas/データフレームをソートする

# インデックス (行名・列名) に基づいてソートする
# sort_index() メソッドを利用して、インデックス（カラム名、行名）に基づいてソートを行うことができます。
# ascending=False は、降順にソートすることを示します。なお、ascending=False を省略すると、
# 昇順でのソートとなります。
# axis=1 が行方向のソートを意味し、省略した場合は、行名に基づくソートとなります。

import pandas as pd
import numpy as np
 
# データフレーム df を作成
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
 
# 行名に基づいてソート
temp = df.sort_index(ascending=False)
print(temp)
 
# カラム名 (列名) に基づいてソート
temp = df.sort_index(axis=1, ascending=False)
print(temp)

# 値に基づいてソートする
# sort_values() メソッドを利用して、データフレームを値に基づいて並び替えを行うことができます。

# B 列の値の小さい順 (昇順) にソート
temp = df.sort_values(by='B') 
print(temp)

# C 列の値の大きい順 (降順) にソート
temp = df.sort_values(by='C', ascending=False)
print(temp)