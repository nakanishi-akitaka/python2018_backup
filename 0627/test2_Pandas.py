# -*- coding: utf-8 -*-
"""
Pandas のデータフレームを確認する
Created on Wed Jun 27 11:21:02 2018

@author: Akitaka
"""

# https://pythondatascience.plavox.info/pandas/データフレームを確認
# 先頭 N 行を表示する
# head([表示する行数]) メソッドでデータフレームの先頭 N 行を切り出すことができます。

import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(6,4), columns=list('ABCD'))
print(df)

#　先頭3行を抽出
print(df.head(3))

# 行数を省略した場合は、千頭5行が抽出されます
print(df.head())

#%%
# 末尾 N 行を表示する
# tail([表示する行数]) メソッドでデータフレームの末尾 N 行を切り出すことができます。

# 末尾 2 行を抽出
print(df.tail(2))

# 行数を省略した場合は、末尾 5 行が抽出されます。
print(df.tail())

#%%
# 基本統計量を算出する
# describe() メソッドをで、件数 (count)、平均値 (mean)、標準偏差 (std)、最小値(min)、
# 第一四分位数 (25%)、中央値 (50%)、第三四分位数 (75%)、最大値 (max) 
# を確認することができます。

print(df.describe())
#%%
# 各列の型を確認する
# 作成したデータフレームの dtypes アトリビュートにアクセスすることで、
# 各列の型 (dtype) を確認することができます。

# データフレーム df2 を作成
df2 = pd.DataFrame({ 'A' : 1.,
                         'B' : pd.Timestamp('20130102'),
                         'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                         'D' : np.array([3] * 4,dtype='int32'),
                         'E' : pd.Categorical(["test","train","test","train"]),
                         'F' : 'foo' })

# データフレーム df2 を表示
print(df2)
print(df2.dtypes)

#%%
'''
列名を表示する
データフレームの列名の一覧を取得するには、columns アトリビュートにアクセスします。
'''
# データフレーム df3 を作成
dates = pd.date_range('20130101', periods=6)
df3 = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
 
# データフレーム df3 を表示
print(df3)
print(df3.columns)

'''
行名 (index) を表示する
データフレームの行名 (インデックス) の一覧を取得するには、index アトリビュートにアクセスします。
'''
print(df3.index)

'''
値のみを 2 次元行列として表示する
value アトリビュートにアクセスすることで、列名、行名を除いた値のみの行列を取得できます。
'''
print(df3.values)

