# -*- coding: utf-8 -*-
"""
scikit-learn で決定木分析 (CART 法)
https://pythondatascience.plavox.info/scikit-learn/scikit-learnで決定木分析
決定木分析 (Decision Tree Analysis) は、機械学習の手法の一つで決定木と呼ばれる、
木を逆にしたようなデータ構造を用いて分類と回帰を行います。
なお、決定木分析は、ノンパラメトリックな教師あり学習に分類されます。

Created on Thu Jun 28 14:32:45 2018

@author: Akitaka

決定木分析の長所
決定木分析は他の分析手法と比較して、以下の特長があります。

* 入力データから特徴を学習し、決定木と呼ばれる樹木状の構造で学習結果を視覚化でき、ルールをシンプルに表現できる特徴があります。
* 他の多くの手法では、データの標準化 (正規化) やダミー変数の作成を必要とするのに対し、決定木分析では、このような前処理の手間がほとんど不要です。
* カテゴリカルデータ (名義尺度の変数) と数値データ (順序尺度、間隔尺度の変数) の両方を扱うことが可能です。
* ニューラルネットなどのようなブラックボックスのモデルと比較して、決定木はホワイトボックスのモデルだといえ、論理的に解釈することが容易です。
* 検定を行って、作成したモデルの正しさを評価することが可能です。


# 主要な決定木分析のアルゴリズム
決定木分析には、いくつかの手法が存在します。各手法の違いについて以下に述べます。

## ID3
ID3 (Iterative Dichotomiser 3) は、1986 年に、Ross Quinlan によって開発されました。カテゴリカル変数に対して情報量を用いて、貪欲な手法 (≒ あるステップにおいて局所的に評価の高いパターンを選択する) で木を構築します。

## C4.5
C4.5 は、ID3 の後継で、数値データの特徴量を動的に離散化するロジックを導入し、全ての特徴量がカテゴリカル変数でなればならないという制約を取り除きました。C4.5 は学習済みの木を if-then で表されるセットに変換し、評価や枝刈り (決定木における不要な部分(枝)を取り除くこと) に用いられます。

## C5.0
C5.0 は C4.5 の改善版で、より少ないメモリ消費で動作するよう、パフォーマンスの改善が行われています。

## CART
CART (Classification and Regression Trees) は、C4.5 によく類似した手法ですが、数値データで表現される目的変数 (=回帰) にも対応している特徴があります。CART は、ルールセットを計算するのではなく、バイナリ・ツリーを構築します。なお、scikit-learn は最適化したバージョンの CART を実装しています。

"""

# scikit-learn に実装されている決定木分析
# それでは、実際にデータを用いてモデルを作成して、その評価を行いましょう。
# scikit-learn では決定木を用いた分類器は、sklearn.tree.DecisionTreeClassifier 
# というクラスで実装されています。

# scikit-learn を用いた決定木の作成
# 今回の分析例では、scikit-learn に付属のデータセット、Iris を利用します。
# このデータセットには、アヤメのがく片や花弁の幅、長さと、そのアヤメの品種が 150 個体分記録されています。
# 今回は、がく片や花弁の幅、長さを説明変数 (教師データ) 、
# アヤメの品種を目的変数 (正解データ) として扱い、分類する決定木を作成します。

# データを読み込み
from sklearn.datasets import load_iris
iris = load_iris()
 
# 説明変数 (それぞれ、がく片や花弁の幅、長さを示します)
print(iris.data)

# 目的変数 (0, 1, 2 がそれぞれの品種を表します)
print(iris.target)

# scikit-learn にて決定木による分類が実装されているクラス、 
# tree.DecisionTreeClassifier クラスの fit メソッドに、
# 説明変数と目的変数の両方を与え、モデル (=決定木) を作成します。
# 今回は木の深さの最大値として、max_depth=3 を指定しています。

# モデルを作成
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(iris.data, iris.target)

# 次に、作成したモデルに説明変数のみを与え、モデルを用いて予測 (分類) を実行します。
# 作成したモデルを用いて予測を実行
predicted = clf.predict(iris.data)

# 予測結果
print(predicted)

# 識別率を確認
print(sum(predicted == iris.target) / len(iris.target))

 #%%
# 決定木の可視化
# 作成した決定木は、DOT ファイルとして、木構造を可視化できるようにエクスポートすることができます。
# 作成した DOT ファイルはオープンソースのグラフ可視化ソフトウェア、 GraphViz を利用して、開くことができます。

# 本例では引数に、各説明変数の名前の指定 feature_names=iris.feature_names 、
# 目的変数の名前の指定class_names=iris.target_names 、
# 枝に色を塗る filled=True 指定と、枝の角を丸めるrounded=True 指定を行っています。
tree.export_graphviz(clf, out_file="tree.dot",
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True)

# DOT ファイルはやや馴染みのないファイル形式かと思いますが、pydotplus パッケージを利用することで、PDF ファイルとして出力することができます。
# 作成した決定木を可視化 (pydotplus パッケージを利用)
import pydotplus
from sklearn.externals.six import StringIO
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

# PDFファイルに出力
# graph.write_pdf("graph.pdf")
# Graphvizをインストールしたが、PDFを直接作るのは無理だった


# このプログラムではGraphvizを使っている
# https://note.mu/univprof/n/n38855bb9bfa8
# 決定木によって得られたルールを見るためには、DTResult.dotというファイルを
# Graphvizというソフトウェア(アプリ)で開く必要があります。
# Graphvizをインストールされていない方は、こちらからダウンロードしてインストールしてください。


