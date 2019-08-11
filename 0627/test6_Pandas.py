# -*- coding: utf-8 -*-
"""
Pandas のデータフレームの行・列の長さを確認する
このページでは、Pandas で作成したデータフレームの行 (レコード)・列 (カラム) のサイズ (大きさ) 
を確認する方法を紹介します。
Created on Wed Jun 27 12:58:11 2018

@author: Akitaka
"""
# # https://pythondatascience.plavox.info/pandas/行・列の長さを確認

'''
行の長さを確認する
index アトリビュートでインデックスの一覧を取得し、len関数でその長さを求めると行の長さが取得できます。
'''
import pandas as pd
import numpy as np
 
# データフレーム df を作成
df = pd.DataFrame(np.random.randn(6, 4), columns=list('ABCD'))
print(df)
 
# 行の長さを取得する
print(len(df.index))

'''
列の長さを確認する
shape アトリビュートでカラム名の一覧を取得し、len 関数でその長さを求めると列の長さが取得できます。
'''

# 列の長さを取得する
print(len(df.columns))

'''
行と列の長さを確認する
shape アトリビュートにアクセスすることで、行と列の長さを配列で取得できます。
'''

# 行と列の長さを取得する
print(df.shape)
