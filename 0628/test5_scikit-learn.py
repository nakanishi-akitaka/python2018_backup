# -*- coding: utf-8 -*-
"""
scikit-learn でクラス分類結果を評価する
https://pythondatascience.plavox.info/scikit-learn/分類結果のモデル評価
本ページでは、Python の機械学習ライブラリの scikit-learn を用いて、
クラス分類 (Classification) を行った際の識別結果 (予測結果) の精度を評価する方法を紹介します。

Created on Thu Jun 28 15:17:40 2018

@author: Akitaka

混同行列 (Confusion matrix)
機械学習を用いたクラス分類の精度を評価するには、
混同行列 (Confusion matrix, 読み方は「コンフュージョン・マトリックス」) を作成して、
正しく識別できた件数、誤って識別した件数を比較することが一般的です。

以下の表は、電子メールのスパムフィルタ (迷惑メールフィルタ) の精度評価を行なう場合の
混同行列の例で説明します。
混同行列は横方向に識別モデルが算出した識別結果、縦に実際の値 (正解データ) を記します。
https://pythondatascience.plavox.info/wp-content/uploads/2017/10/precision1-768x198.png

例えば、スパムフィルタの場合、横方向に「スパム、またはスパムでないとモデルが識別した件数」、
縦方向に「実際にそのメールがスパムであったか、またはスパムでなかったか」を記します。

なお、ブログ記事のカテゴリ分類など、二値分類でない多クラスの分類 (=多値分類) の場合は
以下のような表を作成する場合もあります。
https://pythondatascience.plavox.info/wp-content/uploads/2017/10/precision2-768x259.png

※ 本ページでは、横に識別結果、縦に実際の値を記載しましたが、縦横の記載に決まりはなく、
横に実際の値、縦に識別家結果を記載する場合もあります。

"""
# 混同行列を作成
# scikit-learn には、混同行列を作成するメソッドとして、sklearn.metrics.confusion_matrix があります。
# 以下の例では、スパムフィルタを例に、混同行列を作成します。

from sklearn.metrics import confusion_matrix
y_true = [0, 0, 0, 0, 1, 1, 1, 0, 1, 0]   # 実際の値 (0:スパムでない, 1:スパム)
y_pred = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1]   # 識別結果 (0:スパムでない, 1:スパム)
print(confusion_matrix(y_true, y_pred))

# 多クラスの混同行列
# 多クラスの場合も扱うことができます。以下例では、ブログの記事が、政治 (P: Polictics)、経済 (E, Economics)、スポーツ (S:Sports) のいずれになるかを記事の内容から識別した場合は、以下の例のようになります。
from sklearn.metrics import confusion_matrix
y_true = ["P", "P", "P", "E", "E", "E", "S", "S", "S"] # 実際の分類
y_pred = ["P", "E", "E", "E", "E", "P", "S", "S", "S"] # 機械学習で識別した分類
print(confusion_matrix(y_true, y_pred, labels=["P", "E", "S"]))

# ※ パラメータ labels は、表示する順序を制御します。
# ※ パラメータ labels を指定しなかった場合の並び順は、ABC 順になります。日本語の場合は文字コードの順序によって決定されます。

# 混同行列を作成することで、作成したモデルが実際に正しく識別できている性能を持っているのか、誤って識別されているケースにはどういったケースが多いのかが、一目でわかるようになります。

#%%
# =============================================================================
# 混同行列を解釈するための指標
# 混同行列を解釈するには、次に示すいくつかの指標を用います。
# ここでは、識別結果の正解/不正解で 2 クラスに分類した混同行列の各マスを以下のように呼びます。
# 真陽性 : TP (True-Positive)
# ポジティブに分類すべきアイテムを正しくポジティブに分類できた件数を指します。
# スパムフィルタの場合、「スパム」と分類すべきメールを「スパム」のフォルダに正しく分類できた件数を指します。
#
# 真陰性: TN (True-Negative)
# ネガティブに分類すべきアイテムを正しくネガティブに分類できた件数を指します。
# スパムフィルタの場合、「スパムでない」と分類すべきメールを「スパムでない」のフォルダに正しく分類できた件数を指します。
# 
# 偽陽性: FP (False-Positive)
# ネガティブに分類すべきアイテムを誤ってポジティブに分類した件数を指します。
# スパムフィルタの場合、「スパムでない」と分類すべきメールを「スパム」のフォルダに誤って分類した件数を指します。
# 
# 偽陰性: FN (False-Negative)
# ポジティブに分類すべきアイテムを誤ってネガティブに分類した件数を指します。
# スパムフィルタの場合、「スパム」に分類すべきメールを「スパムでない」のフォルダに誤って分類した件数を指します。
# 
# TP, TN, FP, FN を求める
# scikit-learn では、以下のようなコードで混同行列からそれぞれの値を取得できます。
# =============================================================================
from sklearn.metrics import confusion_matrix
y_true = [0, 0, 0, 0, 1, 1, 1, 0, 1, 0]
y_pred = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1]
tp, fn, fp, tn = confusion_matrix(y_true, y_pred).ravel()
print((tp, fn, fp, tn))

# 識別精度を評価するための指標
# 上記で説明した TP, TN, FP, FN を用いて、識別精度を評価するための指標がいくつかあります。
# 本ページでは、主要な指標のみを紹介します。
# 本ページにない指標は、Wikipedia (英語版) の Confusion matrix – Wikipedia 
# の記事に詳しく記されているので、参考にしましょう。
# 
# 正解率 (Accuracy)
# 正解率 (Accuracy) とは、「本来ポジティブに分類すべきアイテムをポジティブに分類し、
# 本来ネガティブに分類すべきアイテムをネガティブに分類できた割合」を示し、以下の式で表されます。
# Accuracy = (TP + TN) / (TP + TN + FP + FN)
# scikit-learn には sklearn.metrics.accuracy_score として、計算用のメソッドが実装されています。
#
# 精度 (Precision)
# 精度 (Precision) とは、「ポジティブに分類されたアイテムのうち、実際にポジティブであったアイテムの割合」を示し、以下の式で表されます。
# Precision = TP / (TP + FP)
# scikit-learn には sklearn.metrics.precision_score として、計算用のメソッドが実装されています。
# 
# 検出率 (Recall)
# 検出率 (Recall) とは、「本来ポジティブに分類すべきアイテムを、正しくポジティブに分類できたアイテムの割合」を示し、以下の式で表されます。
# 検出率は、真陽性率 (TPR, True-Positive Rate) または、感度 (Sensitivity) とも呼ばれます。
# Recall = TPR = Sensitivity = TP / (TP + FN)
# scikit-learn には sklearn.metrics.recall_score として、計算用のメソッドが実装されています。
# 
# F 値
# F 値 (F-measure, F-score, F1 Score とも呼ばれます) とは、精度 (Precision) 
# と検出率 (Recall) をバランス良く持ち合わせているかを示す指標です。
# つまり、精度は高くても、検出率が低いモデルでないか、
# 逆に、検出率は高くても、精度が低くなっていないか、といった評価を示します。
# F 値は、以下の式のように、検出精度 (Precision) と、検出率 (Recall) の調和平均で求められ、
# 0 〜 1 の間の数値で出力され、0 の場合最も悪い評価、1 の場合最も良い評価となります。
# F1 = 2 * (precision * recall) / (precision + recall)
# scikit-learn には sklearn.metrics.f1_score として、計算用のメソッドが実装されています。

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
y_true = [0, 0, 0, 0, 1, 1, 1, 0, 1, 0]
y_pred = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1]
print(accuracy_score(y_true, y_pred))
print(precision_score(y_true, y_pred))
print(recall_score(y_true, y_pred))
print(f1_score(y_true, y_pred))
