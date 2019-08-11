# -*- coding: utf-8 -*-
"""
Pandas でデータフレームを作ってみよう
Created on Wed Jun 27 11:03:07 2018

@author: Akitaka
"""
# https://pythondatascience.plavox.info/pandas/pandasでデータフレームを作ってみよう
import numpy as np
import pandas as pd

'''
Series (1 次元の値のリスト) を作成する
pd.Series() を用いて、1 次元のリスト (Series, シリーズと呼ばれます) を作成します。
'''
# 数値で構成される Series を作成
s = pd.Series([1,3,5,np.nan,6,8])
print(s)

# 日付の Series を作成
dates = pd.date_range('20130101', periods=6)
print(dates)
#%%

'''
データフレームを作成する
それでは、データフレームを作成してみましょう。本例では、A～Fの各列に数値、文字列、日付、Numpy の行列などを格納します。
'''
df = pd.DataFrame({ 'A' : 1.,
                        'B' : pd.Timestamp('20130102'),
                        'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                        'D' : np.array([3] * 4,dtype='int32'),
                        'E' : pd.Categorical(["test","train","test","train"]),
                        'F' : 'foo' })
print(df)

#%%
'''
Numpy の 行列からデータフレームを作成する
Numpy で作成した行列をデータフレームに変換することもできます。
本例では、 6 x 4 の 2 次元の行列からデータフレームを作成し、
各列に A, B, C, D という名前を付与します。
'''
matrix = np.random.randn(6,4)
print(matrix)

df2 = pd.DataFrame(matrix, columns=list('ABCD'))
print(df2)
#%%
'''
ディクショナリからデータフレームを作成する
Python のディクショナリ (Python 以外のプログラミング言語では
ハッシュまたは連想配列とも呼ばれます) から
データフレームを作成には、from_dict() メソッドを利用します。
'''
import pandas as pd
import numpy as np
a_values = [1, 2, 3, 4, 5]
b_values = np.random.rand(5)
c_values = ["apple", "banana", "strawberry", "peach", "orange"]
my_dict = {"A": a_values, "B": b_values, "C":c_values}
print(my_dict)

my_df = pd.DataFrame.from_dict(my_dict)
print(my_df)
