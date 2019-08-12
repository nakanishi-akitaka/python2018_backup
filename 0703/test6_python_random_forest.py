# -*- coding: utf-8 -*-
"""
【Pythonで決定木 & Random Forest】タイタニックの生存者データを分析してみた
http://www.randpy.tokyo/entry/python_random_forest
Created on Tue Jul  3 14:47:01 2018

@author: Akitaka
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
df = sns.load_dataset("titanic")

# %matplotlib inline
sns.countplot('sex',hue='survived',data=df)

from sklearn.model_selection import train_test_split
#欠損値処理
df['fare'] = df['fare'].fillna(df['fare'].median())
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna('S')

#カテゴリ変数の変換
df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)
df['embarked'] = df['embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
# print(df.columns)
df = df.drop(['class','who','adult_male','deck', 'embark_town','alive','alone'],axis=1)
train_X = df.drop('survived', axis=1)
train_y = df.survived
(train_X, test_X ,train_y, test_y) = train_test_split(train_X, train_y,
 test_size = 0.3, random_state = 666)

print(df.dtypes)
'''
欠損値処理(とりあえず今回は平均値を代入)と、カテゴリカル変数を数値に変換する処理を施しています。
（欠損値処理に関しては、色々な手法があるので、また別の機会に紹介できればと）

最後の行のところで、train_test_splitというライブラリを用いて、学習データと検証データに分割してあげます。
random_stateでseedを固定しないと毎回学習データと検証データが変わってしまうので、固定しときましょう。

さて、分析の前処理（かなり適当ですが…)は終わったので、次項から分析に移っていきます！！
'''

#%%
# 決定木
# scikit-learnの中のライブラリtreeを使っていきます。
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(train_X, train_y)
pred = clf.predict(test_X)
'''
scikit-learnのお馴染みの流れになります。(モデルを定義して、fitで学習)
設定できるパラメータは、例えば以下の様なものです。

criterion : 分割基準。gini or entropyを選択。(デフォルトでジニ係数)
max_depth : 木の深さ。木が深くなるほど過学習し易いので、適当なしきい値を設定してあげる。
max_features：最適な分割をする際の特徴量の数
min_samples_split：分岐する際のサンプル数
random_state：ランダムseedの設定。seedを設定しないと、毎回モデル結果が変わるので注意。
上記以外にも設定できるパラメータがあるので、詳細については、以下公式ドキュメントをご参照ください。
sklearn.tree.DecisionTreeClassifier — scikit-learn 0.19.1 documentation
http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

学習が無事にできたかと思うので、検証データを使って精度を見ていきましょう。

'''

from sklearn.metrics import (roc_curve, auc, accuracy_score)

pred = clf.predict(test_X)
fpr, tpr, thresholds = roc_curve(test_y, pred, pos_label=1)
print(auc(fpr, tpr))
print(accuracy_score(pred, test_y))

'''
sklearn.metricのaucとaccuracy_scoreを使って評価していきます。

AUCは、機械学習でよく使われるモデルの評価指標で、1に近づくほど精度が高いです。
ROC曲線やAUCについては、以下の記事でまとめてあるので是非ご覧ください。
http://www.randpy.tokyo/entry/roc_auc

accuracy_scoreでは、単純な正解率を計算することができます。
計算してみると、AUC:0.784, 正解率:0.795という感じです。
'''

#%%
'''
木構造の可視化
決定木を分析したからには、木の構造を可視化してみましょう。
pythonで可視化するために、以下2つのライブラリをインストールしといてください。
pip install pydotplus
brew install graphviz
以下が可視化するためのコードになります。
'''
#可視化
import pydotplus
from IPython.display import Image
# from graphviz import Digraph
from sklearn.externals.six import StringIO

dot_data = StringIO()
# clf.export_graphviz(clf, out_file=dot_data,feature_names=train_X.columns, max_depth=3)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("graph.pdf")
# Image(graph.create_png())

#%%
'''
scikit-learnのensembleの中のrandom forest classfierを使っていきます。
ちなみに、回帰で使用する場合は、regressionを選択してください。
以下がモデルの学習を行うコードになります。
'''
#ランダムフォレスト
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
clf = clf.fit(train_X, train_y)
pred = clf.predict(test_X)
fpr, tpr, thresholds = roc_curve(test_y, pred, pos_label=1)
print(auc(fpr, tpr))
print(accuracy_score(pred, test_y))

'''
モデルのパラメータとしては、例えば以下のようなものが設定できます。

n_estimators:木をいくつ生成するか。デフォルトでは10。
max_depth:木の深さの設定
max_features:分岐に用いる説明変数の数を設定
min_sample_split:分割する際の最小のサンプル数を設定
random_state:seedの設定。seedを設定しないとモデルが毎回変わるので注意。
こちらも決定木同様、上記以外にも多くのパラメータを設定することができます。
3.2.4.3.1. sklearn.ensemble.RandomForestClassifier — scikit-learn 0.19.1 documentation

精度を算出してみると、
AUC:0.784, 正解率:0.802という結果になりました。

先程の決定木の精度が、AUC：0.784,正解率：0.795でしたので、ほぼほぼ変わらないですね…。
今回、デフォルトのパラメータで学習したので、チューニングを行えば、精度自体はもう少し上がるかと思います。
(ちなみに、パラメータ：n_estimatorsを10→20に変更したとき、AUC：0.804,正解率：0.821)
'''
print('')

#%%
# 変数重要度の可視化
# ランダムフォレストでは、どの変数が重要であったかをfeature_importancesというメソッドを使うことで出すことができます。
# それを図にプロットするコードが以下になります。

import matplotlib.pyplot as plt
# %matplotlib inline

features = train_X.columns
importances = clf.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(6,6))
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.show()