# -*- coding: utf-8 -*-
"""
python機械学習でなぜpandasが利用されているのか
ref:
http://aiweeklynews.com/archives/50579494.html

Created on Mon Jul  9 15:52:30 2018

@author: Akitaka
"""

#pandasをインポート
import pandas as pd

df = pd.DataFrame({
        'A_column' : [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 7, 8, 9, 10],
        'B_column' : [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8]
    })
print(df)

#%%
# ilocを使った列選択（特徴量と教師ラベルを分割して格納）
train_data = df.iloc[:, 0:-1].values
label_data = df.iloc[:, -1].values
# ※csvの右端の列に教師ラベルがあるケースを想定
print(train_data)
print(label_data)

#%%
# loc,ilocを使った列選択（列名指定）
print(df.loc[:,"A_column"] )
print(df.loc[:,["A_column","B_column"]] )
print(df.iloc[:,0:-1] )

#%%
# 列の追加
df['C_column'] = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7, 0, 0, 0, 0, 0, 0]
print(df)

#%%
#カラムAの昇順でソート
print(df.sort_values(by=['A_column'], ascending=True))


#カラムBの降順でソート
print(df.sort_values(by=['B_column'], ascending=False))
