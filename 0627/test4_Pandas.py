# -*- coding: utf-8 -*-
"""
Pandas のデータフレームに行や列 (カラム) を追加する
このページでは、Pandas で作成した、もしくは、読み込んだデータフレームに行や列 (カラム) 
を追加する方法について紹介します。
Created on Wed Jun 27 12:35:49 2018

@author: Akitaka
"""

#https://pythondatascience.plavox.info/pandas/pandasのデータフレームに行や列を追加する

import pandas as pd
import numpy as np
 
# データフレーム df を作成
df = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
print(df)

# データフレーム df2 を作成
df2 = pd.DataFrame([[5, 6]], columns=list('AB'))
print(df2)

# データフレーム df と df2 を結合
df=df.append(df2)
print(df)
'''
複数行の追加は以下のようにして行います。
'''

# データフレーム df3 を作成
df3 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AB'))
print(df3)

# データフレーム df と df3 を結合
df=df.append(df3)

print(df)

# サイトにあるような、df.append(df2)のみ　ではダメだった

'''
上記の例では、元の行番号のまま追加が行われますが、
ignore_index=True パラメータを指定することで、新たな行番号を割り当てることができます。
'''
# データフレーム df と df3 を結合 (元の行番号を無視)
dft = df.append(df3, ignore_index=True)
print(dft)

#%%
'''
列 (カラム) を追加する
作成済みのデータフレームに新しい列名を追加することで、列の追加ができます。
追加するデータは Python のリストや Numpy の行列 (Array) を指定できます。
'''
# データフレーム df を作成
df = pd.DataFrame([["0001", "John"], ["0002", "Lily"]], columns=['id', 'name'])
print(df)
 
# 列 "job" を追加
df['job'] = ["Engineer", "Sales"]
print(df)

# 列 "age" を追加 (Numpy Array を追加)
df['age'] = np.array([35, 25])
print(df)

