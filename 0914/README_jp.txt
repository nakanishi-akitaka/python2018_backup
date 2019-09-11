# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 15:34:30 2018

@author: Akitaka
"""

[1b1] 半教師あり学習昨日の補足
> [?] 圧力について内側に補完する分にはいいかもしれない。確実に存在はする。
>     物質によって、データ数が偏るのを少しは緩和できるかもしれない
と書いたが、どうやらそうはいかないらしい。
調べてみたところ、LPもLSもクラス分類の手法なので、圧力について数値を補完することは難しい。


[1b2] 半教師あり学習のテスト
[todo] LP(orLS)の半教師あり学習
サンプルを実行
Decision boundary of label propagation versus SVM on the Iris dataset
http://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_versus_svm_iris.html
.fit()と.predict()を使う分には、今までのと同様っぽい

Label Propagation learning a complex structure
http://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_structure.html
label_spread.transduction_
label_spread.predict(X)
この2つは多分同じ事をやろうとしている？
どちらも[0,...,0,1,...,1](0*100, 1*100)という結果になる
このサンプルでは、
X = 円2つのxy座標
y = 0(外側) or 1(内側)
labels = 0,1がそれぞれ一点のみ、他はすべて-1(教師なしデータ)
として.fit(X,labels)を実行している。
その結果、.predict(X) でyを再現できている。

http://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html
labels = iris.targetに-1を30%ほど混ぜた
.fit(iris.data, labels)の学習後、
label_prop_model.transduction_
label_prop_model.predict(iris.data)
のどちらでも再現できている

ref:20180912
LPもLSもscikit-learnにある。半教師あり学習はこの2つのみ
http://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelPropagation.html
http://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html
ドキュメントによる説明
http://scikit-learn.org/stable/modules/label_propagation.html
    教師なしデータは、y=-1を入力しておく！



[1b3] 半教師あり学習のテスト２
[todo] LP(orLS)+SVMの半教師あり学習
http://univprof.com/archives/17-02-28-12926352.html
http://univprof.com/archives/17-02-25-12818365.html
LSでYの値のないサンプルにYの値を割り振ってから、すべてサンプルを使ってSVMモデルを構築します。
    1.クロスバリデーションを行い、LSのα・γを選ぶ
    2.LSによりクラスの情報がないサンプルにクラスを割り振る
    3.クロスバリデーションを行い、SVMのC・γを選ぶ
    4.SVMモデル構築
LSでYの値のないサンプルも有効に活用して、SVMにより高性能なモデルを構築することで、
良好なクラス分類結果が得られると期待できます。
※LS,LPだけでもクラス分類はできる
　しかし、クラス分類には、性能の高いサポートベクターマシンを使いたいから組み合わせる

https://datachemeng.com/semisupervisedlearningmerit/
ここでいう、
1.クラス間の境界が明確になる (クラス分類のみ)
が可能になる
プログラムとしては、上のサンプルの後に普通にSVMやればそれでいいハズ
クロスバリデーションはないけれど

作成した
./test4_lp_svm.py


[1c] 今後の方針
[todo] 日物のQ&Aを読んで、次の仕事を考える
今後の展開は？
説明変数を三元系でも応用できるものに変えていく
    LaH10が出る様なものはあるか？
    質量・体積などは説明変数にしないのか？
    第一原理計算も、原子番号・個数・圧力からTc決める云々
        フェルミ面の状態密度・空間群などを説明変数にして予測しても、実験で検証する時には意味がない
    水素比率が重要そうな気がする。LaH10のTcより。
    デバイ振動数などは説明変数に入らないのか？
        超伝導のAD式に使うパラメータωlog,λを使えば100%再現するが、それでは意味がない。
    ランダムフォレストで特徴量選択する
    説明変数を熱伝導などに変える場合、「圧力かけた状態での」値であるべきだから計算する
    結晶構造は分からないのか？(説明変数にしないのか？)
他の計算手法を試す
    どういうときにどの手法がいいという経験則はあるのか？
        始めたばかりなのでよく分からない。SVMやRFは比較的よく使われるが。
データの数増やす
    VH3とかは期待できるのか？
    実は先行研究があったことに昨日きづいた。外れてた。
    機械学習の検証にはなったと言える。
