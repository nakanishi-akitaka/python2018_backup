# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 11:50:19 2018

@author: Akitaka
"""

[1e3a]
One-class SVM with non-linear kernel (RBF)
http://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html
example:
test1_OCSVM.py
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
.fit()がXのみ＝教師なし学習
この場合、クラスタリングして、普通の値は+1.異常なものは-1を値として返す。
例えばサンプルだと、
X_train = [[ 2.28213613  2.13803064]. [ 2.51128174  1.21860675]....]
y_pred _train = [ 1 -1 ... ]


[1e3b]
Anomaly detection with Local Outlier Factor (LOF)
http://scikit-learn.org/stable/auto_examples/neighbors/plot_lof.html
example:
test2_LOF.py
    y_pred = clf.fit_predict(X)
    y_pred_outliers = y_pred[200:]
test1では、X_train = 正常値のみ, X_outliers = 異常値ばかり(ただの乱数)だが、
ここでは、X = 正常値 + 異常値(乱数)となっている
また、predictをやってはいるが、実はこの後でy_predは使わない。.fit()で充分。


[1e3c]
Outlier detection with several methods.
http://scikit-learn.org/stable/auto_examples/covariance/plot_outlier_detection.html
example:
test3_OCSVM_EE_IF_LOF.py
contamination =　外れ値の割合(デフォルトで0.1), OCSVMだとnuが関係する
正規分布から外れると、EEは上手くいかない。他３つはOK。


[1e4] SVM + OCSVM
http://univprof.com/archives/17-01-28-11535179.html
SVM・SVRでクラス分類モデル・回帰モデルをつくるときは、OCSVMでモデルの適用領域を決めるのがよい
SVM・SVR＋OCSVMは以下の手順で行います。
1.クラス分類問題であればSVMモデル、回帰問題であればSVRモデルをつくる 
2.OCSVMモデルをつくる 
  OCSVMモデルをつくるとき、SVM・SVRと同じカーネル関数・カーネル関数のパラメータを使います。
  たとえば、SVM・SVRでガウシアンカーネルを使ったときは、OCSVMでもガウシアンカーネルを使い、
  SVM・SVRで最適化されたγの値をOCSVMでも使います。
以下、[新しいデータの推定] 
3.OCSVMモデルにデータを入力して、モデルの適用範囲の中か外かを判定する
4.モデルの適用範囲の中のデータをSVMもしくはSVRモデルに入力して目的変数の値を推定する

example:
test4_SVM_OCSVM.py
data: make_classificationで作成したデータを9:1に分ける
予測値、真の値、OCSVMでの検証結果(異常値なら-1)は以下の通り
[[ 1  1  1]
 [ 1  0  1]
 [ 0  0 -1]
 [ 1  1  1]
 [ 0  0  1]
 [ 0  1  1]
 [ 0  0  1]
 [ 0  1 -1]
 [ 0  1  1]
 [ 1  1  1]]
異常値でも合うことはあるし、正常値でも外すことはある。

更に改良
1.testデータを正常値、異常値に分ける
2.正常値、異常値、それぞれの正解不正解の数を表記

Detailed classification report:
             precision    recall  f1-score   support
          0       0.83      0.80      0.81        96
          1       0.82      0.85      0.83       104
avg / total       0.83      0.82      0.82       200
[[77 19]
 [16 88]]

Inlier  sample, number of good/bad predictions: 28 137
Outlier sample, number of good/bad predictions: 7 28
↑を何度か繰り返したものの、InlierとOutlierの差はそこまで極端なものにはならない
おそらく、サンプルデータがそこまでADを外れていないから、正解率が大差ない

SVCの精度が悪い？
標準化をする　→　大差ない
n_clusters_per_class=10 → 1
　trainの正解率100%だが、testの正解率0%に！

ref:
https://note.nkmk.me/python-pandas-len-shape-size/
https://pythondatascience.plavox.info/pandas/行・列の抽出
https://datachemeng.com/ocsvm/




[1f] 論文追試？
https://arxiv.org/pdf/1803.10260.pdf
ref:07/05,06,10,11,12
example:
test5_Eg_RFC_RFR.py




[1g] ダブルクロスバリデーション/二重交差検証/DCV + 21種類のクラス分類方法テスト(総当たり)
ref:0706, 0707,07/11,12
[1g1] 復習
example:
test6_7clf_dcv.py