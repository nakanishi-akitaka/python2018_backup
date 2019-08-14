# -*- coding: utf-8 -*-
"""
実務で使うとこだけ！python機械学習(AI)のデータ処理(pandas/scikit-learn)
ref:
http://aiweeklynews.com/archives/49945455.html

Created on Mon Jul  9 16:02:40 2018

@author: Akitaka
"""
#pandasのインポート
import pandas as pd

#サンプルデータ作成
df = pd.DataFrame({'age' : [33, 25, 52], 
                   'height' : [175, 170, 'NaN'],
                   'weight' : [70, 'NaN' , 60], 
                   'job' : ['employee', 'neet', 'employee']})

df.to_csv('temp.csv')
df = pd.read_csv('temp.csv')
print(df.isnull().sum())

print(df)

#欠損値を含む行を削除
t=df.dropna()
print(t)

#欠損値を含む列を削除
print(df.dropna(axis=1) )

#特定の列に欠損値が含まれている行だけを削除
print(df.dropna(subset=['age']))

#%%
#カテゴリデータを整数に変換
df['job'] = df['job'].map({'employee':1 , 'neet':0})

#sklearnのImputerクラスのインポート
from sklearn.preprocessing import Imputer

#欠測値のインスタンスを作成
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)

#データを適合 
imr = imr.fit(df)

#平均値で補完を実行 
imputed_data = imr.transform(df.values)
print(imputed_data)

#matplotlibをインポート
import matplotlib.pyplot as plt

#グラフを表示
df.plot()

#%%
#平均値
print(df.mean())

#中央値
print(df.median())

#最大値
print(max(df['age']))

#最小値
print(min(df['age']))

#%%
# ilocを使った列選択
print(df.iloc[:,0])  # 番号で選択
print(df.iloc[:,0:2]) #複数で連番も可能

# ixを使った列選択
print(df.ix[:,"age"] )
print(df.ix[:,["age","job"]] )
print(df.ix[:,0:2] )
    
# 列の追加
df['home']=[1,1,0]
print(df)

#%%
#標準化クラスをインポート
from sklearn.preprocessing import StandardScaler

#ST = StandardScaler()
#df = ST.fit_transform(df)

# 行と行の相関係数を表示します。
print(df.corr())

#特徴量データ
X = df.iloc[:, 0:2].values

#出力データ
y = df.iloc[:,2].values

print(X)
print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
