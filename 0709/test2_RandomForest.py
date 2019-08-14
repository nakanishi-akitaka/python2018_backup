# -*- coding: utf-8 -*-
"""
ランダムフォレストで特徴量の重要度を評価する
ref:
http://aiweeklynews.com/archives/50653819.html
Created on Mon Jul  9 15:05:01 2018

@author: Akitaka
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Pandas のデータフレームとして表示
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

#データの定義
x = iris.data
y = iris.target

#学習データとテストデータに分割
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# 学習
clf = RandomForestClassifier(n_estimators=20, random_state=42)
clf.fit(x_train, y_train)

#予測データ作成
y_predict = clf.predict(x_train)

#正解率
print(accuracy_score(y_train, y_predict))

#特徴量の重要度
feature = clf.feature_importances_

#特徴量の重要度を上から順に出力する
f = pd.DataFrame({'number': range(0, len(feature)),
             'feature': feature[:]})
f2 = f.sort_values('feature',ascending=False)
f3 = f2.loc[:, 'number']

#特徴量の名前
label = df.columns[0:]

#特徴量の重要度順（降順）
indices = np.argsort(feature)[::-1]

for i in range(len(feature)):
    print(str(i + 1) + "   " + str(label[indices[i]])
    + "   " + str(feature[indices[i]]))

plt.title('Feature Importance')
plt.bar(range(len(feature)),feature[indices], color='lightblue', align='center')
plt.xticks(range(len(feature)), label[indices], rotation=90)
plt.xlim([-1, len(feature)])
plt.tight_layout()
plt.show()