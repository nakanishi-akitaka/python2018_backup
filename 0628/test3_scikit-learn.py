# -*- coding: utf-8 -*-
"""
scikit-learn でクラスタ分析 (K-means 法)
https://pythondatascience.plavox.info/scikit-learn/クラスタ分析-k-means
本ページでは、Python の機械学習ライブラリの scikit-learn を用いてクラスタ分析を行う手順を紹介します。

Created on Thu Jun 28 14:17:09 2018

@author: Akitaka

クラスタ分析とは
クラスタ分析 (クラスタリング, Clustering) とは、ラベル付けがなされていないデータに対して、
近しい属性を持つデータをグループ化する手法です。例をあげると、以下のような活用方法があり、
マーケティング施策や商品の企画開発などに活用することます。

* 製品ごとの特徴 (自動車であれば、価格や定員、燃費、排気量、直近の販売台数) を用いて類似の製品をグループ化
* 店舗の特徴 (スーパーであれば、売上や面積、従業員数、来客数、駐車場の数) から類似の店舗をグループ化
* 顧客の特徴 (銀行であれば、性別、年齢、貯蓄残高、毎月の支出、住宅ローンの利用有無など)
  を用いて似たような利用傾向の顧客をグループ化
クラスタ分析には大別して、K-Means に代表される「非階層的クラスタ分析」と Ward 法 (ウォード法)
に代表される「階層的クラスタリング」の2種類が存在します。
本ページでは、非階層的クラスタ分析の代表例であるK-Means法を用いたクラスタリングについて解説します。

非階層的クラスタリング
非階層的クラスタリング (例: K-Means 法) では、決められたクラスタ数にしたがって、
近い属性のデータをグループ化します。
以下の図では、3つのクラスタに分類しましたが、それぞれの色でどのクラスタに分類されたかを示しています。

階層的クラスタリング
階層的クラスタリング (例: Ward 法) では、クラスタリングの結果を木構造で出力する特徴があります。
縦方向の長さ (深さ) は類似度を示し、長いほど類似度が低く、短いほど類似度が高いことを示します。

K-Means法とは
K-Means 法 (K-平均法ともいいます) は、基本的には、以下の 3 つの手順でクラスタリングを行います。

1.初期値となる重心点をサンプルデータ (データセット全体からランダムに集めた少量のデータ) から決定。
2.各サンプルから最も近い距離にある重心点を計算によって求め、クラスタを構成。
3.2.で求めたクラスタごとに重心を求め、2. を再度実行する。
  2. ～ 3. を決められた回数繰り返し実行し、大きな変化がなくなるまで計算。

"""

# scikit-learn を用いたクラスタ分析の実行例
# scikit-learn を用いてクラスタ分析を行う手順を紹介します。

# 今回使用するデータ
# 今回は、UC バークレー大学の UCI Machine Leaning Repository にて公開されている、
# 「Wholesale customers Data Set (卸売業者の顧客データ)」を利用します。
# 
# https://archive.ics.uci.edu/ml/datasets/Wholesale+customers
# http://pythondatascience.plavox.info/wp-content/uploads/2016/05/Wholesale_customers_data.csv

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
 
# データセットを読み込み
cust_df = pd.read_csv("http://pythondatascience.plavox.info/wp-content/uploads/2016/05/Wholesale_customers_data.csv")
 
# 不要なカラムを削除
del(cust_df['Channel'])
del(cust_df['Region'])
print(cust_df) 

# Pandas のデータフレームから Numpy の行列 (Array) に変換
cust_array = np.array([cust_df['Fresh'].tolist(),
                   cust_df['Milk'].tolist(),
                   cust_df['Grocery'].tolist(),
                   cust_df['Frozen'].tolist(),
                   cust_df['Milk'].tolist(),
                   cust_df['Detergents_Paper'].tolist(),
                   cust_df['Delicassen'].tolist()
                   ], np.int32)
 
# 行列を転置
cust_array = cust_array.T
 
# クラスタ分析を実行 (クラスタ数=4)
pred = KMeans(n_clusters=4).fit_predict(cust_array)
print(pred)

#%%
# 各クラスタの特徴を確認
# クラスタ分析の結果を利用し、各クラスタがどのような特徴があるのかを確認します。
# ここでは、集計作業を楽に行うため、Pandas のデータフレームを利用します。

# Pandas のデータフレームにクラスタ番号を追加
cust_df['cluster_id']=pred
print(cust_df)

# 各クラスタに属するサンプル数の分布
print(cust_df['cluster_id'].value_counts())

# 各クラスタの各部門商品の購買額の平均値
print(cust_df[cust_df['cluster_id']==0].mean()) # クラスタ番号 = 0
print(cust_df[cust_df['cluster_id']==1].mean()) # クラスタ番号 = 1
print(cust_df[cust_df['cluster_id']==2].mean()) # クラスタ番号 = 2
print(cust_df[cust_df['cluster_id']==3].mean()) # クラスタ番号 = 3

#%%
# Matplotlib でクラスタの傾向を可視化
# 先ほど求めた、各クラスタの各部門商品の購買額の平均値を 
# Matplotlib を用いて傾向を可視化すると以下のようになります。

# Matplotlib で積み上げ棒グラフを出力
# 可視化（積み上げ棒グラフ）
# import matplotlib.pyplot as plt
# タイトルに反して、matplotlibは使っていない
# pandas DataFrameのplot属性を使っている 

clusterinfo = pd.DataFrame()
for i in range(4):
    clusterinfo['cluster' + str(i)] = cust_df[cust_df['cluster_id'] == i].mean()
clusterinfo = clusterinfo.drop('cluster_id')
 
my_plot = clusterinfo.T.plot(kind='bar', stacked=True, title="Mean Value of 4 Clusters")
my_plot.set_xticklabels(my_plot.xaxis.get_majorticklabels(), rotation=0)

# 結果から、それぞれ次のように説明できます。

# クラスタ番号 = 0 に分類された顧客 (79 人) は、Grocery (食料雑貨品) と Detergents_Paper (衛生用品と紙類) の購買額が比較的高いことがわかります。
# クラスタ番号 = 1 に分類された顧客 (291 人) は、全体的に購買額が低い傾向にあります。
# クラスタ番号 = 2 に分類された顧客 (7 人) は、全てのジャンルで購買額が高いと言えます。
# クラスタ番号 = 3 に分類された顧客 (63 人) は、Fresh (生鮮食品) やFrozen (冷凍食品) の購買額が比較的高いことがわかります。
# 上記のように、クラスタ分析は簡単にデータのみからあらゆる発見を行うことに適している汎用的な手法だと言えます。皆さんが会社や研究で扱っているデータもこのように分析することで、新たな発見があるかもしれないでしょう。

