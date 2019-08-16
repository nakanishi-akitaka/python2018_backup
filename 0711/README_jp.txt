# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 10:27:55 2018

@author: Akitaka
"""

[1g] ダブルクロスバリデーション/二重交差検証/DCV + 21種類のクラス分類方法テスト(総当たり)
[1g1] DCVの復習
ref:0706/test4.py, 0707test1_dcv.py
example:
test1_dcv.py


[1g2] 21種類のクラス分類方法テスト(総当たり)の復習(Eg >0 か Eg = 0の分類)
0625/test5.py TMXO2 Eg Accurary のみ
0626/test7.py TMXO2 Eg Accurary, FPR, TPR
0702/test1.py TMXO2 Eg Accuracy, Precision, Recall, F1-score, FPR, TPR
0703/test3.py TMXO2 Eg Accuracy, Precision, Recall, F1-score, AUC
example:
test2_21clf.py
data: iris
score: Accuracy, Precision, Recall, F1-score, FPR, TPR
全部APRF1=1,FPR=0なのであまり意味ない？

[1g4] データセットをirisからmake_classification dataset に変えて[1g2]をリトライ
example:
test2_21clf.py


[1g3] 分類データセットの作成関数 make_classification のテスト
Plot randomly generated classification dataset
http://scikit-learn.org/stable/auto_examples/datasets/plot_random_dataset.html
example:
test3_plot_clf_data.py

公式ドキュメントの説明
http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
その和訳？とテスト
http://overlap.hatenablog.jp/entry/2015/10/08/022246
example:
test4_make_sample.py


[1g5] DCV＋複数モデルでモデルごとの評価
※21個の分類手法を試していたが、多すぎるので、抜粋する
論文を参考にする
Yingchun Cai et al., J. Chem. Inf. Model.
DOI: 10.1021/acs.jcim.7b00656
ref:06/05,22,28
クラス分類手法は７種類で検討。
 decision tree (DT), k-nearest neighbors (k-NN), logistic regression (LR),
 naive Bayes (NB), neural network (NN), random forest (RF), and
 support vector machine (SVM)

モデルごとにハイパーパラメータが違うので、
dcv(mod,param_grid)
のように二重交差検証を関数、モデルとハイパーパラメータをインプットとした
NBは最適化するべきハイパーパラメータがないに等しいので、飛ばす

３重入れ子の辞書を作って、まとめることに成功
range_d = np.arange( 1, 10, dtype=int)
param_grid = [{'max_depth': range_d}]
test = {
        'DT':{'name':'Decision Tree','model':dtc,'param':param_grid},
        'RF':{'name':'Random Forest','model':dtc,'param':param_grid},
        }
for key, value in test.items():
    print(test[key]['name'])
    dcv(test[key]['model'],test[key]['param'])

とりあえず、
NB, ハイパーパラメータなし
NN, ハイパーパラメータ多すぎる
の２つは除いた５つで一括DCVに成功
example:
test5_7clf_dcv.py


[1f2] EgでRFC+RFRのテスト
example:
test6_Eg_RFC_RFR



