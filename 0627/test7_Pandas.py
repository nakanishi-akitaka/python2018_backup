# -*- coding: utf-8 -*-
"""
Pandas のデータフレームの行⇔列を入れ替える
このページでは、Pandas のデータフレームの行と列を転置する方法について紹介します。
Created on Wed Jun 27 13:01:14 2018

@author: Akitaka
"""

# https://pythondatascience.plavox.info/pandas/データフレームの転置
# 行⇔列を転置する
# データフレームの T アトリビュートにアクセスすると、データフレームの縦、横を入れ替えたデータフレームを
# 取得できます。なお、T は Transpose の頭文字です。

import pandas as pd
import numpy as np

# データフレーム df を作成
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
print(df)
 
# データフレーム df を転置
print(df.T)