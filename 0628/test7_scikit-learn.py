# -*- coding: utf-8 -*-
"""
scikit-learn に付属しているデータセット
https://pythondatascience.plavox.info/scikit-learn/scikit-learnに付属しているデータセット
scikit-learn には、機械学習やデータマイニングをすぐに試すことができるよう、実験用データが同梱されています。
このページでは、いくつかのデータセットについて紹介します。

Created on Thu Jun 28 15:55:41 2018

@author: Akitaka
"""

'''
Iris (アヤメの計測データ、通称：アイリス)
“setosa”, “versicolor”, “virginica” という 3 種類の品種のアヤメの
がく片 (Sepal)、花弁 (Petal) の幅および長さを計測したデータです。

データセットの詳細

レコード数	150
カラム数	4
主な用途	分類 (Classification)
データセットの詳細	UCI Machine Learning Repository: Iris Data Set
各カラムの構成

sepal length (cm)	がく片の長さ
sepal width (cm)	がく片の幅
petal length (cm)	花弁の長さ
petal width (cm)	花弁の幅
'''

# データセットを読み込み
from sklearn.datasets import load_iris
iris = load_iris()
 
# Pandas のデータフレームとして表示
import pandas as pd
print(pd.DataFrame(iris.data, columns=iris.feature_names))

# 各データの分類 (0: 'setosa', 1: 'versicolor', 2: 'virginica')
print(iris.target)
print(iris.target_names)

#%%
'''
Boston house-prices (ボストン市の住宅価格)
米国ボストン市郊外における地域別の住宅価格のデータセット。

データセットの詳細

レコード数	506
カラム数	14
主な用途	回帰 (Regression)
データセットの詳細	UCI Machine Learning Repository: Housing Data Set
各カラムの構成

CRIM	人口 1 人当たりの犯罪発生数
ZN	25,000 平方フィート以上の住居区画の占める割合
INDUS	小売業以外の商業が占める面積の割合
CHAS	チャールズ川によるダミー変数 (1: 川の周辺, 0: それ以外)
NOX	NOx の濃度
RM	住居の平均部屋数
AGE	1940 年より前に建てられた物件の割合
DIS	5 つのボストン市の雇用施設からの距離 (重み付け済)
RAD	環状高速道路へのアクセスしやすさ
TAX	$10,000 ドルあたりの不動産税率の総計
PTRATIO	町毎の児童と教師の比率
B	町毎の黒人 (Bk) の比率を次の式で表したもの。 1000(Bk – 0.63)^2
LSTAT	給与の低い職業に従事する人口の割合 (%)

'''

# データセットを読み込み
from sklearn.datasets import load_boston
boston = load_boston()
 
# 説明変数を Pandas のデータフレームとして表示
import pandas as pd
print(pd.DataFrame(boston.data, columns=boston.feature_names))

# 目的変数 (1,000 ドル台でオーナーが所有する住宅の価格の中央値)
print(boston.target)

#%%
'''
Diabetes (糖尿病患者の診断データ)
糖尿病患者 442 人の検査数値と 1 年後の疾患進行状況（正規化済み）。

データセットの詳細

レコード数	442
カラム数	10
主な用途	回帰 (Regression)
データセットの詳細	Least Angle Regression diabetes.sdata.txt

''' 

from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
 
# Pandas のデータフレームとして表示
import pandas as pd
print(pd.DataFrame(diabetes.data, columns=("age", "sex", "bmi", "map", "tc", "ldl", "hdl", "tch", "ltg", "glu")))
# 目的変数 (1 年後の疾患進行状況)
print(diabetes.target)
 
#%%
'''
Digits (数字の手書き文字)
0 ～ 9 の 10 文字の手書きの数字を 64 (8×8) 個の画素に分解したものです。

データセットの詳細

レコード数	1,797 (10 クラスの場合)
カラム数	64
主な用途	分類 (classification)
データセットの詳細	UCI Machine Learning Repository: Optical Recognition of Handwritten Digits Data Set
'''
from sklearn.datasets import load_digits
digits = load_digits(n_class=10)
 
# データセットの概要
print(digits.DESCR)

# Pandas のデータフレームとして表示
import pandas as pd
print(pd.DataFrame(digits.data))

# 目的変数 (手書きの内容)
print(digits.target_names)

print(digits.target)

#%%
'''
Linnerud (生理学的特徴と運動能力の関係)
ノースカロライナ州立大学の A. C. linnerud 博士が作成した、20 人の成人男性に対してフィットネスクラブで測定した 3 つの生理学的特徴と 3 つの運動能力の関係。

データセットの詳細

レコード数	20
カラム数	説明変数:3, 目的変数: 3
主な用途	多変数回帰 (multivariate regression)
データセットの詳細	R: Linnerud Dataset
説明変数の構成

Weight	体重
Waist	ウエスト (胴囲)
Pulse	脈拍
目的変数の構成

Chins	懸垂の回数
Situps	腹筋の回数
Jumps	跳躍
'''

from sklearn.datasets import load_linnerud
linnerud = load_linnerud()
 
# データセットの概要
print(linnerud.DESCR)

# Pandas のデータフレームとして表示
import pandas as pd
print(pd.DataFrame(linnerud.data, columns=linnerud.feature_names))

# 目的変数 (3 種類) を Pandas のデータフレームとして表示
import pandas as pd
print(pd.DataFrame(linnerud.target, columns=linnerud.target_names))
