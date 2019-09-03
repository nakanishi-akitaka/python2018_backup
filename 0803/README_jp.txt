# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 10:58:18 2018

@author: Akitaka
"""

[1b2]
tipsに追加
回帰モデル・クラス分類モデルを使うときに必ずやらなければならないたった１つのこと
～モデルを適用できるデータ領域(適用領域・適用範囲)の設定～
http://univprof.com/archives/16-05-30-3588574.html
ADの決め方４つ
    1.各変数 (特徴量・記述子・説明変数・入力変数) について、99.7%のサンプルが含まれる範囲をADとする。
        変数が多くなると、計算が煩雑で、かつ外れが多くなる。
    2.データセットの中心からの距離について、99.7%のサンプルが含まれるものをADとする。
        すべての変数を用いて距離を計算するので、1.のような欠点はないが、
        変数の間に非線形性があったり、データの分布のかたまり(クラスター)が複数にあると使えない。
    3.データの密度について、99.7%のサンプルが含まれるものをADとする。
        サンプルごとに最も近い距離にあるいくつかのサンプルとの距離の平均の逆数, OCSVMなど
    4.アンサンブル学習において、複数の予測値の標準偏差が小さいものをADとする。

金子研究室のサイトより
モデルを作るのにサンプル数はいくつ必要か？に対する回答～モデルの適用範囲・モデルの適用領域～
https://datachemeng.com/numberofsamplesad/
上と同様に、1.2.3を紹介(4.のみなし)
1.2は欠点(同上)があることから、オススメしているのは3.
データ密度の計算方法
    最も距離の近いk個のサンプルとの距離の平均の小ささ
    OCSVM
モデルの適用範囲・モデルの適用領域 (Applicability Domain, AD) 
～回帰モデル・クラス分類モデルを使うとき必須となる概念～
https://datachemeng.com/applicabilitydomain/
上と同様に1.2.3.4を紹介


[!?] k-NNの場合は、ADとは別に信頼度がある。
AD＝データ密度
信頼度＝周りのサンプルの値の標準偏差
アンサンブル学習だと、AD＝各モデルの推定値の標準偏差が小さいもの、となる。
ref:
金子研究室のサイトでも同様
https://datachemeng.com/ensemblelearning/




[1c] 06/15 (Anaconda使い始め)からの総括　追加
arxiv.1803.10260

0705test3_atomic_data.py
原子ごとの物理量データベースを作成する
arxiv.1803.10260.csv
原子ごとの物理量データベース
欠損値をチェック。
https://en.wikipedia.org/wiki/Electron_affinity
電子親和力は一部がない。周期表の終わりの方はともかく、Mgですらない！



0711test5_7clf_dcv.py
[1g] ダブルクロスバリデーション/二重交差検証/DCV + 7種類のクラス分類方法テスト(総当たり)
ref:
0706/test4.py
0707/test1_dcv.py
0707DCVの目的(回帰係数？推定性能？モデル選択？)やDCV予測値について詳しい
0711/test1_dcv.py
0711/test5_7clf_dcv.py
モデルごとに違うハイパーパラメータで最適化
[1g5] DCV＋複数モデルでモデルごとの評価
    21個の分類手法を試していたが、多すぎるので、論文を参考に抜粋する
    モデルごとにハイパーパラメータが違う
    ３重入れ子の辞書を作って、まとめることに成功
→DCVと合流



0711test2_21clf.py
[1g2] 21種類のクラス分類方法テスト(総当たり)
ref:
0625/test5.py
0626/test7.py
0702/test1.py
0703/test3.py
0711/test2_21clf.py
0330/test2.py
0613/test0.py






0713test5_Eg_RFC_RFR.py
分類+回帰
[1f2] EgでRFC+RFRのテスト
ref:
0705,0706,
0711/test6_Eg_RFC_RFR.py
0712/test1_Eg_RFC_RFR.py
0713/test5_Eg_RFC_RFR.py




test0_SVM_OCSVM_DCV_clf.py
test0_SVM_OCSVM_DCV_rgr.py
SVM+OCSVMの開発
ref:
0713, 0716, 0717, 0718, 0724, ...
→DCVと合流
0726test0_SVM_OCSVM_contour.py
等高線を使った図示はここで終了

test0_kNN_AD_DCV_clf.py
test0_kNN_AD_DCV_clf.py
ref:
0726, 0727


0521test2_convert_cf2parameter.py　→　削除(test1_to_csv.pyに組み込んだため)

test1_to_csv.py
ref:
0521/test2.py tc.csvファイルで化学式を原子番号と個数に変換
0521/test3.py SVMでtc予測
0724/test2_Tc_SVM.pyで逆変換
0724 csvに出力

test2_Tc_kNN_AD_DCV.py
ref:
0727, ...

test2_Tc_SVM_OCSVM_DCV.py
ref:
0426/test1.py が重要
    cvの数、cv=KFold, Shufflesplitなどの違いに言及
    Shufflesplitがベストと結論
    また、ShuffleSplitとscalingを併用することで性能が向上したとも書いてある
0427/test1.py
    スケーリングをMinMax, Standardのどちらにするかの比較
0511/test1.py = 0427/test1.py
    発表前に軽く計算しただけ
07/18 水素化物Tc計算
07/24以降 アプデ



test0_{clf,rgr}.pyを削除。
test0_SVM_OCSVM_{clf,rgr}.pyの後半を省略したものに過ぎない。



[1d] 
？？？　StandardScaler()の位置は、train, test分割の前後どっち？
→後！
X_train　.fit_transform(X_train)
X_test　.fit(X_train) -> .transform(X_test)
でなければならない

プログラムは、以下の通り
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=...)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
パイプラインやPCAを使う場合
    std_clf = make_pipeline(StandardScaler(), PCA(n_components=2), GaussianNB())
    std_clf.fit(X_train, y_train)
    pred_test_std = std_clf.predict(X_test)
    scaler = std_clf.named_steps['standardscaler']
    X_train_std = pca_std.transform(scaler.transform(X_train))


ref:
http://scikit-learn.org/stable/auto_examples/applications/plot_prediction_latency.html
http://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html
http://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html
https://qiita.com/makopo/items/35c103e2df2e282f839a 
https://qiita.com/ishizakiiii/items/0650723cc2b4eef2c1cf
http://datanerd.hateblo.jp/entry/2017/09/15/160742
https://ohke.hateblo.jp/entry/2017/08/11/230000
https://dev.classmethod.jp/machine-learning/introduction-scikit-learn/
https://rindalog.blogspot.com/2018/03/grid-searching-pipeline.html

逆の場合もある？ミス？
http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
http://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html


[1d2] StandardScaler, pipelineの処理を確認
example:
test0_scaler.py
  .fit(X_train) + .transform(X_train) + .transform(X_test)
= .fit_transform(X_train) + .transform(X_test)

pipelinで、scalerのtransformを省略できる！
scaler = StandardScaler()
X_train1 = scaler.fit_transform(X_train)
X_test1 = scaler.transform(X_test)
model = SVR()
model.fit(X_train1,y_train)
y_pred1 = model.predict(X_test1)
→
pipe = make_pipeline(StandardScaler(),SVR())
pipe.fit(X_train,y_train)
y_pred2 = pipe.predict(X_test)

print(np.allclose(y_pred1,y_pred2)) -> True




[1e] jupyter notebook
jupyterで実験ノートも書き込む方がいい？
http://www.procrasist.com/entry/2016/10/20/200000
https://pythondatascience.plavox.info/pythonの開発環境/jupyter-notebookを使ってみよう
example:
test0_jupyter.ipynb




[1f]
回帰分析の表示をかえたように、二値分類についても回帰分類と同様に
計算値、CV、test(例のnoteでは予測用)の予測値、およびそれらについての
TP, FP, FN, TNと正解率(マルチクラスは正解率のみ)を計算する。
関数を完成させた！
ただし、もう一度CVをやる関係上、時間はかかるようになった。

ref:
0731 [1a3] 回帰計算の結果はどう表示するべき？
0731/test0_rgr.py
0731/mylibrary.py
https://note.mu/univprof/n/n38855bb9bfa8
■StatisticsAll.csv ・・・ 
それぞれのクラス分類手法におけるモデル構築用データ・クロスバリデーション・予測用データ1の
それぞれTrue Positive・False Positive・False Negative・True Negative・正解率

example:
test0_clf.py
my_library.py




[1g] jobスクリプト改良
※機械学習関係ないけど、わざわざ別の項目つくるほどでもないのでここへ
~/tools-6.0/0scf
~/tools-6.2/0scf
において、
mv /work/nakanishi/${JOB_ID}  ${SGE_CWD_PATH}
を使用する方向に変更。
エラーが出た場合、
mv /work/nakanishi/${JOB_ID}  ${SGE_CWD_PATH}/${JOB_ID}_ng
とする。

