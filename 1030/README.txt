# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:52:31 2018

@author: Akitaka
"""

[1b] サイトで勉強2 note.nkmk.me
Python関連記事まとめ - note.nkmk.me
https://note.nkmk.me/python-post-summary/
数値
    小数・整数を四捨五入するroundとDecimal.quantize
    ランダムな整数を生成するrandom.randrange, randint
    割り算の商と余りを同時に取得するdivmod
    数値の整数部と小数部を同時に取得するmath.modf
    数値が整数か小数かを判定

算数・数学
    Pythonで算数・数学の問題を解く
    最大公約数と最小公倍数を算出・取得
    fractionsで分数（有理数）の計算
    complex型で複素数を扱う（絶対値、偏角、極座標変換など）
    set型で集合演算（和集合、積集合や部分集合の判定など）
    階乗、順列・組み合わせを計算、生成
    指数関数・対数関数を計算（exp, log, log10, log2）
    三角関数を計算（sin, cos, tan, arcsin, arccos, arctan）
    平均、中央値、最頻値、分散、標準偏差を算出

以上のページのサンプルプログラムを写経(コピペ)完了


https://note.nkmk.me/python-complex/
> 平方根は**0.5でも算出できるが、誤差が生じる。cmath.sqrt()を使うと正確な値が算出できる。
複素数の話だが、実数でも？

https://note.nkmk.me/python-math-exp-log/
> 複素数を扱う場合、**演算子を使った例では誤差が生じているが、cmathモジュールを使うとより正確な値が得られる。
> math.exp(x)はmath.e ** xと等価でなく、math.exp(x)のほうがより正確な値となる。
> 常用対数（10を底とする対数）は、math.log10(x)で計算できる。math.log(x, 10)よりも正確な値となる。
> 二進対数（2を底とする対数）は、math.log2(x)で計算できる。math.log(x, 2)よりも正確な値となる。
細かく専用の関数が用意されているのは、正確さの為らしい




[1c] 書籍(の英語サイト)で勉強 Python Data Science Handbook
https://jakevdp.github.io/PythonDataScienceHandbook/

What Is Machine Learning?
https://jakevdp.github.io/PythonDataScienceHandbook/05.01-what-is-machine-learning.html

Introducing Scikit-Learn
https://jakevdp.github.io/PythonDataScienceHandbook/05.02-introducing-scikit-learn.html
> Because it is so fast and has no hyperparameters to choose, 
> Gaussian naive Bayes is often a good model to use as a baseline classification,
> before exploring whether improvements can be found through more sophisticated models.
ガウシアンナイーブベイズは、テストとして実行するのには良いらしい

> Here we will use a powerful clustering method called a Gaussian mixture model (GMM),
> discussed in more detail in In Depth: Gaussian Mixture Models.
ガウス混合モデルによるクラスタリングは強力らしい

Hyperparameters and Model Validation
https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html
検証曲線：モデルの複雑さ(多項式の次数やハイパーパラメータ)vsスコア
    トレーニングスコアは単調減少
    バリデーションスコアは凸
    バリデーションスコアが最大になるのが最適
学習曲線：データ数vsスコア
    トレーニングスコアは単調減少
    バリデーションスコアは単調増加
    両者が十分収束するだけのデータ数があればいい
見るべきハイパーパラメータの数が増えると、視覚化できない
→ハイパーパラメータのグリッドサーチ

Feature Engineering
https://jakevdp.github.io/PythonDataScienceHandbook/05.04-feature-engineering.html
特徴量の扱い方
単純にラベル→1,2,3と数値化するのはまずいのでワンホットエンコードする
文章のベクトル化
モデルの選択ではなく、特徴量変換によって、性能を向上させる
欠損データ処理


以上のページのサンプルプログラムを写経(コピペ)完了
※一部変更
https://jakevdp.github.io/PythonDataScienceHandbook/05.02-introducing-scikit-learn.html
from sklearn.cross_validation import train_test_split
->
from sklearn.model_selection import train_test_split

from sklearn.mixture import GMM      # 1. Choose the model class
->
from sklearn.mixture import GaussianMixture      # 1. Choose the model class

cmap=plt.cm.get_cmap('spectral', 10))
->
cmap=plt.cm.get_cmap('nipy_spectral_r', 10))

https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html
from sklearn.learning_curve import validation_curve
->
from sklearn.model_selection import validation_curve

from sklearn.grid_search import GridSearchCV
->
from sklearn.model_selection import GridSearchCV



