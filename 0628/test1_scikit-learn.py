# -*- coding: utf-8 -*-
"""
scikit-learn でトレーニングデータとテストデータを作成する
https://pythondatascience.plavox.info/scikit-learn/トレーニングデータとテストデータ
本ページでは、Python の機械学習ライブラリの scikit-learn を用いて
トレーニングデータとテストデータを作成するために、サンプリングを行なう手順を紹介します。

Created on Thu Jun 28 13:19:49 2018

@author: Akitaka


トレーニングデータ・テストデータとは
教師あり機械学習（回帰分析、決定木分析、ランダムフォレスト法、ナイーブベイズ法、
ニューラルネットワークなど）によるモデルを作成するには、
準備したデータセットをトレーニングデータ（訓練用データ、学習用データとも呼ばれます）と
テストデータ（検証用データ、評価用データ、検証用データとも呼ばれます）の 
2 つに分割して予測モデルの作成、評価を行なうことが一般的です。
このように一定の割合でトレーニングデータとテストデータに分割することを
ホールドアウト (hold-out) と呼びます。

以下は、クレジットカードの解約予測の分析テーマを例に挙げて、そのイメージを説明します。

# トレーニングデータとテストデータの分割
データセット全体 (20 レコード) を本例では、80 : 20 の割合で
トレーニングデータ (16 件) とテストデータ (4件) に分割します。

トレーニングデータとテストデータはどのような割合 (何対何) で分割すべきといった決まりはありませんが、
トレーニングデータ : テストデータを 80 % : 20 % や、75 % : 25 % 、70 % : 30 % の比率で
分割することが一般的です。

# トレーニングデータを用いた機械学習モデルの作成
分割したデータのうち、トレーニングデータのみを用いて、
説明変数 (x) と目的変数 (y) の関係性を学習し、
説明変数 (x) が与えられたときに、目的変数 (y) を返す機械学習モデルを作成します。

# テストデータを用いた予測の実行
続いて、作成した機械学習モデルとテストデータの説明変数 (x) のみを用いて、予測結果を算出します。

# テストデータを用いた評価
前段で求めた予測結果と、実際の解約有無を比較することで、
どれだけ正確に予測できるかを確認することで、機械学習モデルの予測性能を測ります。

# サンプリングを行なうときに注意すべきこと
データセット全体からレーニングデータとテストデータを分割する際に、
データの特性に偏りのあるトレーニングデータやテストデータを使って機械学習モデルを作成すると、
精度の悪いモデルができてしまいます。
それを防ぐために、ランダムに並び替えたデータからデータセットを抽出します。
そのような作業をサンプリング、特に、ランダムに抽出することをランダムサンプリングと呼びます。

"""

# train_test_split: トレーニングデータとテストデータを分割
# scikit-learn には、トレーニングデータとテストデータの分割を行なうメソッドとして
# sklearn.model_selection.train_test_split が用意されています。
# このメソッドは、与えられたデータフレームから、指定された条件に基づいて
# トレーニングデータとテストデータを分割します。

# train_test_split の使用例
# 今回使用するデータフレーム (4 カラム、12 レコード) を作成します

import pandas as pd
from sklearn.model_selection import train_test_split

namelist = pd.DataFrame({
   "name" : ["Suzuki", "Tanaka", "Yamada", "Watanabe", "Yamamoto",
             "Okada", "Ueda", "Inoue", "Hayashi", "Sato",
             "Hirayama", "Shimada"],
   "age": [30, 40, 55, 29, 41, 28, 42, 24, 33, 39, 49, 53],
   "department": ["HR", "Legal", "IT", "HR", "HR", "IT",
                  "Legal", "Legal", "IT", "HR", "Legal", "Legal"],
   "attendance": [1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1]
})
print(namelist)

# テストデータを 30% (test_size=0.3) としてトレーニングデータ、テストデータに分割します。
namelist_train, namelist_test = train_test_split(namelist, test_size=0.3)
print(namelist_train)
print(namelist_test)

#%%
# テストデータを具体的な数値で、5件(test_size=5)としてトレーニングデータ、テストデータに分割します。
namelist_train, namelist_test = train_test_split(namelist, test_size=5)
print(namelist_train)
print(namelist_test)

#%%
# トレーニングデータを 50% (training_size=0.5) としてトレーニングデータ、テストデータに分割します。
namelist_train, namelist_test = train_test_split(namelist,test_size=None, train_size=0.5)
print(namelist_train)
print(namelist_test)

#%%
# データの並び替え（シャッフル）を行わないで、分割のみを実施します。
namelist_train, namelist_test = train_test_split(namelist, shuffle=False)
print(namelist_train)
print(namelist_test)

#%%
# 乱数のシードを42に固定します。1回目と2回目で全く同じサンプリングがなされていることが見て取れます。
namelist_train, namelist_test = train_test_split(namelist, random_state=42)
print(namelist_train)
print(namelist_test)

namelist_train, namelist_test = train_test_split(namelist, random_state=42)
print(namelist_train)
print(namelist_test)

#%%
# “department” のクラスによる層化サンプリングを行います。
#
# 層化サンプリングとは、サンプリングしたデータが偏らないよう、
# 指定した変数の出現頻度が一定になるように調整した上で、サンプリングを行なうものです。
# 以下の例では、”department” (=部門) を stratify=namelist['department'] 
# として指定しているので、テストデータには、
# 各部門 (IT (情報システム), HR (人事), Legal (法務)) が、
# 全体の分布と同じになるよう、各 1 件ずつ抽出されています。
namelist_train, namelist_test = train_test_split(namelist, stratify=namelist['department'])
print(namelist_train)
print(namelist_test)

#%%
# 上記で説明した層化サンプリングを “attendance” (総会への出席状況) 
# の列に基づいて実施した例は以下になります。
# テストデータには、1 が 2 件、0 が 1 件と、全体の分布とほぼ同じように抽出されていることがわかります。
namelist_train, namelist_test = train_test_split(namelist, stratify=namelist['attendance'])
print(namelist_train)
print(namelist_test)

#%%
# 例えば、説明変数 x (attendance 以外) と 目的変数 y (attendance) を分割し、
# train_test_split に 2 つ以上の引数を与えることもできます。
# 以下の例では、データフレームとarray を渡し、データフレーム、array を 2 つずつ返します。

# データセットを説明変数と目的変数に分割
namelist2_x = namelist.drop("attendance", axis=1)
namelist2_y = namelist['attendance']
 
# 説明変数
print(namelist2_x)

# 目的変数
print(namelist2_y)

#%%
# サンプリングを実施し、トレーニングデータ、テストデータに分割します。
x_train, x_test, y_train, y_test = train_test_split(namelist2_x, namelist2_y, test_size=0.3)
# 説明変数 (トレーニングデータ)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

