# -*- coding: utf-8 -*-
"""
ROC曲線とAUCについて定義と関係性をまとめたよ

Created on Wed Jun 27 20:47:18 2018

@author: Akitaka
"""
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
import numpy as np
from sklearn.metrics import roc_auc_score
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
print(roc_auc_score(y_true, y_scores))

# https://qiita.com/koyamauchi/items/a2ed9f638b51f3b22cd6
# Scikit-learnでAUCを計算する
# roc_auc_score()に、正解ラベルと予測スコアを渡すとAUCを計算してくれます。
# 楽チンです。
# →　エラー

# 正解率とAUCを計算して最適なモデルを選択するスクリプト
# かなり冗長だが、学習過程で作ったコードを貼ってみました。
# import basice apis
import numpy as np
import pandas as pd
# %matplotlib inline  
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  
from sklearn.metrics import roc_auc_score

# import Sample Data to learn models
dataset = load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.DataFrame(dataset.target, columns=['y'])

# cross-validation by holdout
X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y,
                                                 test_size=0.20,
                                                 random_state=1)
# set pipelines for two different algorithms
pretrained_pipes = []
trained_pipes = []
pipe_knn = Pipeline([('scl',StandardScaler()),('est',KNeighborsClassifier())])
pipe_logistic = Pipeline([('scl',StandardScaler()),('est',LogisticRegression())])
pipe_gbc = Pipeline([('scl',StandardScaler()),('est',GradientBoostingClassifier())])

pretrained_pipes.append(pipe_knn)
pretrained_pipes.append(pipe_logistic)
pretrained_pipes.append(pipe_gbc)

# パイプラインの学習
for pipeline in pretrained_pipes:
    trained_pipe = pipeline.fit(X_train,y_train.as_matrix().ravel())
    trained_pipes.append(trained_pipe)

# パイプラインの評価（評価は指定指標の下で実施されるようにすること）
# 結果格納データフレーム生成用に各種配列を作成
result_clumns = ['name','accurate_rate','roc']
result_names = ['KNN','LOGISTIC','GBC']
result_accuracy = []
result_roc = []

# 各モデルで性能評価する
for pipeline in trained_pipes:
    result_accuracy.append(accuracy_score(y_test,pipeline.predict(X_test)))
    result_roc.append(roc_auc_score(y_test,pipeline.predict(X_test)))

# 　リスト->ディクショナリ->データフレームに変換
values = [result_names,result_accuracy,result_roc]
result_dataframe = pd.DataFrame(dict(zip(result_clumns,values))).loc[:,['name','accurate_rate','roc']]
high_accurate_model = result_dataframe.sort_values(by=["accurate_rate"], ascending=False).iloc[0,[0]].values[0]
high_accurate_score = result_dataframe.sort_values(by=["accurate_rate"], ascending=False).iloc[0,[1]].values[0]
high_roc_model = result_dataframe.sort_values(by=["roc"], ascending=False).iloc[0,[0]].values[0]
high_roc_score = result_dataframe.sort_values(by=["roc"], ascending=False).iloc[0,[2]].values[0]

result_dataframe

#結果呼び出し用関数
def model_selection(test_score):
    if test_score == 'accurate':
        print('最も正解率が高かったのは',high_accurate_model,'で、その値は',round(high_accurate_score,4),'でした')
    elif test_score == 'auc':
        print('最もAUCが高かったのは',high_roc_model,'で、その値は',round(high_roc_score,4),'でした')
        result_dataframe
    else:
        print('エラー！model_selection関数には、auc か accurateを引数として渡してください。')

#　関数呼び出し
model_selection('accurate')
model_selection('auc')
model_selection('hogehoge')
