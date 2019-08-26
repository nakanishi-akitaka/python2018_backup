# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 11:16:42 2018

@author: Akitaka
"""


[1a3] PCAの寄与率は計算できる？
http://scikit-learn.org/0.18/modules/generated/sklearn.decomposition.PCA.html
属性:explained_variance_ratio_ でできるっぽい
example:
test0_PCA.py


[1c] Tc SVM
[1c1] test0_rgr.py改良
yyplotの部分をモジュール化 -> my_libraryに追加
ref:
0427/test1.py


[1c2] パラメーター変換について 
example:
test1_to_csv.py

ref:
0521/test2.py

化学式　→　原子番号＋個数だけでなく、
原子番号＋個数　→　化学式をテスト

from  pymatgen import periodic_table 
for i in range(1,20):
    print(periodic_table.get_el_sp(i))
番号　→　元素記号がついに分かった
※今までのは少しインチキ

material.reduced_formula
を使えば、S1H3 → H3S　と表示される

csvファイルへの出力方法を、pandasのto_csvを使ったものに変更
また、インデックスと列の名前を追加
これに伴い、Tc計算する方も、csvの読み込み方法を変更
example:
test2_Tc_SVM.py


[1c3] Tc予測
example:
test2_Tc_SVM.py

更新の余地リストから、以下の内容をアップデート
* 予測データをDataFrameにして、csvに出力 Tcの高さでソート
* 化学式を一まとめの文字列に変換する　出力形式でそれっぽく見せるのはやめる


ref:
pandasの使い方
https://note.nkmk.me/python-pandas-read-csv-tsv/
https://note.nkmk.me/python-pandas-to-csv/
https://note.nkmk.me/python-pandas-sort-values-sort-index/




[1d1] SVM+OCSVM+ダブルクロスバリデーション/二重交差検証/DCV
ref:
0713, 0716, 0717,0718
http://univprof.com/archives/16-06-12-3889388.html
https://datachemeng.com/doublecrossvalidation/

example:
test3_SVM_OCSVM_DCV_clf.py

DCVを10回行う
  ave, std of accuracy of inner CV: 0.930 (+/-0.036)
  ave, std of accuracy of inner CV: 0.918 (+/-0.036)
  ave, std of accuracy of inner CV: 0.920 (+/-0.032)
  ave, std of accuracy of inner CV: 0.910 (+/-0.004)
  ave, std of accuracy of inner CV: 0.924 (+/-0.024)
  ave, std of accuracy of inner CV: 0.916 (+/-0.000)
  ave, std of accuracy of inner CV: 0.928 (+/-0.032)
  ave, std of accuracy of inner CV: 0.930 (+/-0.012)
  ave, std of accuracy of inner CV: 0.914 (+/-0.036)
  ave, std of accuracy of inner CV: 0.926 (+/-0.020)
  予測性能は比較的安定していると言える

この後、いままで通りのSVM+OCSVMを実行する
続けてやる意味あんまりない？


[1d2] DCVは繰り返しやるもの
http://univprof.com/archives/16-06-12-3889388.html
ダブルクロスバリデーションを複数回繰り返すことで (たとえば100回繰り返すことで)、
どれくらいr2DCV・RMSEDCV・正解率DCVにばらつきがあるのかを検討することが重要

https://datachemeng.com/doublecrossvalidation/
ダブルクロスバリデーションにおける外側のクロスバリデーションの推定値と実測値との
比較を行うことで、モデルの推定精度を検証する
ダブルクロスバリデーションを何回か行って、
その結果の平均やばらつきを確認するとより細かくモデルを検証できます。


[1d3] DCVの意義
https://datachemeng.com/modelvalidation/
https://datachemeng.com/doublecrossvalidation/
データ数が大　→　トレーニング + バリデーション + テスト
データ数が中　→　クロスバリデーション + テスト
データ数が小　→　ダブルクロスバリデーション
CVは2,5,10-foldが一般的

あくまでも推定性能の検証に用いるもの！
train_test_splitと併用するのはおかしい！？


