# -*- coding: utf-8 -*-
"""
Pandas のデータフレームの特定の行・列を削除する

このページでは、Pandas で作成したデータフレームの特定の行 (レコード) 、列 (カラム) 
を除去・取り除く方法について紹介します。
なお、条件に基づいて特定の行や列を抽出する方法については、
「Pandas でデータフレームから特定の行・列を取得する」もご覧ください。

Created on Wed Jun 27 12:54:23 2018

@author: Akitaka
"""
# https://pythondatascience.plavox.info/pandas/行・列を削除
'''
特定の行を削除する
DataFrame.drop() メソッドを利用して、インデックスに基づいて特定の行の削除を行うことができます。
リストを指定して、複数の行を一度に削除することもできます。
'''

import pandas as pd
import numpy as np
 
# データフレーム df を作成
df = pd.DataFrame(np.random.randn(6,4), columns=list('ABCD'))
print(df)
 
# 行 5 を削除
temp = df.drop(5)
print(temp)

# 行 3 と 4 を削除
temp = df.drop([3,4])
print(temp)

#%%

'''
特定の列を削除する
列の削除は行と同様に、DataFrame.drop() メソッドを利用しますが、
引数に、axis=1 を指定し、列の削除であることを指定します。
'''

# 列 A を削除
temp = df.drop("A", axis=1)
print(temp)

# より簡単な方法として、Python の del ステートメントを利用する方法もあります。

# 列 A を削除
del df['A']
print(df)
