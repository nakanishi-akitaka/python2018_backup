# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 10:26:33 2018

@author: Akitaka
"""

* test1_web.py
* test2_web.py
* neet_app.html
* neetHello.html
Pythonで人工知能のWebサービスを実装する方法
http://aiweeklynews.com/archives/48462559.html

* 618551/test0.job
jobのテスト

* test3_DL.py
scikit-learnのディープラーニング実装簡単すぎワロタ
http://aiweeklynews.com/archives/50172518.html

* test4_questionnaire.py
人工知能(AI)を自社に導入したい人はscikit-learnを利用しよう
http://aiweeklynews.com/archives/49372407.html







[1d] サイトで勉強４　新規事業のつくり方
カテゴリ： Pythonで機械学習
http://aiweeklynews.com/archives/cat_1301202.html

下をやってみる　できればグーグルサイトで？
Pythonで人工知能のWebサービスを実装する方法
http://aiweeklynews.com/archives/48462559.html
インストール
$ conda install flask
$ conda install wtforms
example:
test1_web.py
test2_web.py
neet_app.html
neetHello.html
　上手くいかず

機械学習（AI）勉強の最初のステップはPythonに触れる事です
http://aiweeklynews.com/archives/50420363.html
anaconda, spyder推奨
もしも本ブログを参考に機械学習を勉強していく場合、以下の順番がおすすめです。
STEP1　pythonを理解する　   ：今読んでいる記事
STEP2　機械学習を理解する　：人工知能(AI)入門　～機械学習でできることを解説～
STEP3　scikit-learnを実装する：これだけは知っておけ！PythonでＡＩ開発の基礎まとめ
STEP4　データ処理を実装する：python機械学習(AI)のデータ処理(pandas/scikit-learn)
STEP5　Webサービスを作る　：Pythonで人工知能のWebサービスを実装する方法
まあ機械学習を一言でいってしまうと、データから関数f(x)を作成して評価してるだけなんですけどね。


交差検定（クロスバリデーション）など機械学習の評価方法まとめ
http://aiweeklynews.com/archives/50219779.html
「ホールドアウト法」と「交差検定（クロスバリデーション）」と「混合行列」が分かれば大丈夫
1.ホールドアウト法
 モデルを作る学習データと、モデルを評価するテストデータに分割して評価
2.交差検定（クロスバリデーション）
 省略
3.混合行列
  データに偏りがある場合、正解率だけではモデルの良し悪しが分からない
（参考）ホールドアウト法と交差検定はどちらが良いのか
交差検定とホールドアウト法を比べた場合、交差検定の方がより評価結果の信頼が高くなります。


scikit-learnのディープラーニング実装簡単すぎワロタ
http://aiweeklynews.com/archives/50172518.html
画像認識分野で非常に高い性能を発揮します。
理由は、畳み込みニューラルネットワークとReLU関数です。
畳み込みニューラルネットワークにより、画像の特徴を適切に捉えることが出来るようになりました。
ReLU関数では、誤差逆伝播法で微分が１になることから、勾配が消えなくなるという問題を解決しました。
またそもそも計算量が低いため、活性化関数として非常に利用されています。
ディープラーニングが流行った理由は、特徴量を自動抽出できること
  ディープラーニングの精度が他のアルゴリズムに優れているかどうかはまた別問題
  データが少ないとまともな精度が出ない
・教師データが１万件を超えてくると精度が高くなってくる。
・教師データが少ないと、非常に精度が低くなる。分類問題で20%の精度とか普通にある。鞍点に落ちていると思われる。
・特徴量の数を変えて実験したが精度が安定しない。SVMやランダムフォレストはやはり安定感あるので、実務ではこっちを使いたくなる。
・ディープラーニングは中のモデルがブラックボックスになるので、結果に対して説明できないのは辛い。
・ディープラーニングはパラメータの数が多すぎ。潤沢な分析環境は必須。
・画像とテキストデータ以外でのディープラーニングの活用は少し早いかも。

example:
test3_DL.py  
サンプルにかいてある、csvファイルが用意されてないので、書き換え
しかし、学習がうまく収束しない！原因は不明

ref:
http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
http://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html
API Reference
http://scikit-learn.org/stable/modules/classes.html
siciki-learnにある関数のまとめ？

実務で使うとこだけ！python機械学習(AI)のデータ処理(pandas/scikit-learn)
http://aiweeklynews.com/archives/49945455.html

人工知能(AI)を自社に導入したい人はscikit-learnを利用しよう
http://aiweeklynews.com/archives/49372407.html
人工知能って要は"分類する"ということが本質なんです。
example:
test4_questionnaire.py


これだけは知っておけ！PythonでＡＩ開発の基礎まとめ
http://aiweeklynews.com/archives/48508096.html

人工知能(AI)入門　～機械学習でできることを解説～
http://aiweeklynews.com/archives/49438967.html


[1d2] キーワード検索
畳み込みニューラルネットワーク(CNN), ReLU関数(活性化関数の１つ)
多層パーセプトロンの実装
scikit-learnのMLPClassifierおよびMLPRegressor
http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
サンプルなし！

http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
ここから、サンプル実行

example:
http://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_training_curves.html
test5_DL.py

http://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html
test6_DL.py

http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
test7_DL.py

もう一つはエラーでたので消した




[1e] サイトで勉強５　大学教授のブログ (データ分析相談所)
http://univprof.com/

カテゴリ：データ解析
http://univprof.com/archives/cat_128443.html

以下は、手法の説明というよりは、自作プログラムの宣伝
http://univprof.com/archives/17-03-03-13051684.html
http://univprof.com/archives/17-03-02-13011545.html
http://univprof.com/archives/17-03-01-12967159.html
http://univprof.com/archives/17-02-28-12926352.html
http://univprof.com/archives/16-05-28-3515242.html
http://univprof.com/archives/17-02-25-12818365.html
http://univprof.com/archives/17-02-21-12639621.html

重要な説明変数(記述子・特徴量・入力変数)を選択するには、複数の方法で選択された結果が必要
http://univprof.com/archives/17-02-09-12105097.html
Q.変数選択の方法はたくさんありますが、その中のどれを使ったほうがいい?
A.多くの手法を使うしかない
  あるデータセットがあるとき、各手法のアルゴリズム・評価方法・評価値が
  そのデータセットに適しているかどうかは、誰も分かりません。
  いくつかの変数選択手法を実行して、どんな変数が何回選ばれたかを確認する

[プログラム・コード公開] コピペだけで１２の変数選択手法を一気に実行して結果を比較する！ (Python言語)
https://note.mu/univprof/n/n8e468337b5f5




[1f] 論文追試？
https://arxiv.org/pdf/1803.10260.pdf
[1f0] 前回まで(07/05)
Table 1,2から、80個の説明変数を作成するプログラム開発
Egでデータベースを作成
各元素のデータベースを作成
ハイパーパラメータ探索
Eg >= 0 で学習　→　小さいギャップのみ完全再現
Eg >  0 で学習　→　まばら

[1f1] 復習
元素ごとのデータベースの作成(欠損値はそのまま)
* test8_makeDB.py
* arxiv.1803.10260.csv
ref:0705test3.py

Egから学習
* test9_Eg.py (Eg=0も使用)
* test10_Eg_nonzero.py (Eg>0のみ使用)
ref:0706test1.py
ref:0706test2.py

[1f2] EgでRFC+RFRのテスト
* test11_Eg_RFC_RFR.py

# 1. Classification: metal (Eg = 0) or insulator (Eg > 0)
#    RandomForestClassifier with default hyper parameters

train data: RMSE, MAE, RMSE/MAE, R^2 = 0.145, 0.021, 6.887, 0.915
0.24 seconds 

Detailed classification report:
The model is trained on the full development set.
The scores are computed on the full evaluation set.

             precision    recall  f1-score   support

        0.0       0.96      0.99      0.98       890
        1.0       0.99      0.97      0.98      1102

avg / total       0.98      0.98      0.98      1992

[[ 882    8]
 [  34 1068]]

# 4. Regression: Energy gap　(Eg > 0)
#    RandomForestRegressor with default hyper parameters

train data: RMSE, MAE, RMSE/MAE, R^2 = 0.344, 0.172, 2.001, 0.952
test  data: RMSE, MAE, RMSE/MAE, R^2 = 0.344, 0.172, 2.001, 0.952
1.15 seconds 

まずまずの精度
※ここでは、全部のデータを分類、Eg>0のみのデータを回帰に使っている
つまり、絶縁体なのに金属と誤分類されたものでも、回帰している

？　Eg = 0をtestデータとすると、Egの予想値は？
train data: RMSE, MAE, RMSE/MAE, R^2 = 0.341, 0.177, 1.924, 0.953
test  data: RMSE, MAE, RMSE/MAE, R^2 = 0.587, 0.280, 2.094, 0.000
yy-plotを見ると、最大で Eg(予測) = 4eV にまで誤差が！
平均的には十分なんだろうけど一部がおかしい

以前の発想
> エネルギーギャップの回帰分析は、変数変換するといいのかも？
> Eg → log(Eg)とする？　Eg = 0　→ log(Eg) = -1000にでもしておく
> しかし、予測値Egがいくら以下なら、金属判定になる？
これよりは分類＋回帰であるべきか





[1g] jobスクリプトのテスト
2018/0312/test6/job
2018/0424/test2.job
2018/0425/test1.job
2018/0426/test1.job
2018/0426/test2.job
2018/0427/test2.job
あたりを参考に
example:
618551/test0.job

これを使えば、二重交差検証をするjobをたくさん投げることで、並列化が可能！

