# -*- coding: utf-8 -*-
"""
Pandas でデータフレームから特定の行・列を取得する
このページでは、Pandas で作成したデータフレームを操作して、
特定の行・列を取得し、目的の形へ加工する手順について紹介します。

なお、特定の行、列を除外する方法については、
「Pandas のデータフレームの特定の行・列を削除する」の記事をご参照ください。
Created on Wed Jun 27 12:03:04 2018

@author: Akitaka
"""

# https://pythondatascience.plavox.info/pandas/行・列の抽出

'''
特定の列を取得する
カラム名 (列名) を指定して、特定の列を抽出できます。
'''
import pandas as pd
import numpy as np
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))

# "A" 列を抽出する
print(df['A'])

# . (ドット) を利用しても同じ結果が得られます。
print(df.A)

#%%
'''
特定の区間の行を抽出する
: (コロン) で行番号を指定することで、特定の区間の行を抽出できます
。行番は 0 行目から始まる点に注意しましょう。
'''
# 1 行目から 3 行目を抽出
print(df[1:3])

# 先頭から 3 行目までを抽出
print(df[:3])

# "20130102" から "20130104" の行を抽出
print(df['20130102':'20130104'])

#%%
'''
loc アトリビュートを使って特定の行・列を抽出する
loc アトリビュートを利用して、ラベルに基づいて特定の行や抽出できます。
'''
# 行名が "2013-01-01" の列を抽出
print(df.loc["2013-01-01"])

# 列 "A", "B" の 2 列を抽出
print(df.loc[:,['A','B']])

# 行名 = "20130102" ～ "20130104" の "A" 列と "B" 列を取得
print(df.loc['20130102':'20130104',['A','B']])

# 行名 = "20130102" の "A" 列と "B" 列を取得
print(df.loc['20130102',['A','B']])


#%%
'''
行や列の位置を指定して行・列を取得する
iloc アトリビュートを用いて、行や列の位置に基づいて行・列を取得することができます。
行や列は 0 行目・0列目から始まる点に注意しましょう。
'''
# 3 行目を取得
print(df.iloc[3])

# 1,2,4 行目と 0-2 列目を取得
print(df.iloc[[1,2,4],[0,2]])

# 1-3 行目と全ての列を取得
print(df.iloc[1:3,:])

# 全ての行と 1-3 列目を取得
print(df.iloc[:,1:3])

# 1 行目の 1 列目の値を取得
print(df.iloc[1,1])

#%%
'''
条件を指定して行・列を取得する
True または False を返す式を指定することで、特定の条件式に基づく行・列の取得ができます。
'''

# "A" 列の値が 0 より大きい行を取得
print(df[df.A > 0])

# 値が 0 より大きい値のみを取得
print(df[df > 0])

'''
isin() メソッドと組み合わせて使うことで、複数の特定の値を含む行・列の抽出ができます。
'''
# データフレーム df2 を作成
df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
print(df2)

# "E" 列に "two" または "four" を値に持つ行を抽出
print(df2[df2['E'].isin(['two','four'])])
