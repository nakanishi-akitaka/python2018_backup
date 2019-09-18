# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 14:05:41 2018

@author: Akitaka
"""
[1a2] 水素化物のでもy-randomization ref:20181009,11
前回のTc計算
C:\Users\Akitaka\Downloads\python\0907
ここからコピペして実行 + mylibraryは更新したもの
※データベースの更新については無視

C:\Users\Akitaka\Downloads\python\1012\kNN_y_randamization.ipynb
にてモジュール化

Tc計算でy-randamizationをテスト
→
kNN,RF,SVRの３つとも、明らかに精度が落ちているので、偶然による相関は小さい

kNN
C:  RMSE, MAE, R^2 = 14.488, 8.648, 0.864
CV: RMSE, MAE, R^2 = 25.024, 15.781, 0.595
P:  RMSE, MAE, R^2 = 34.647, 22.440, 0.000

y-randomization
In 10 iterations, average +/- standard deviation
RMSE: 37.198 (+/-0.666)
MAE: 29.239 (+/-0.633)
R^2: 0.105 (+/-0.032)
23.20 seconds 

RF
C:  RMSE, MAE, R^2 = 9.041, 5.963, 0.947
CV: RMSE, MAE, R^2 = 19.768, 13.114, 0.747
P:  RMSE, MAE, R^2 = 43.940, 38.891, 0.000

y-randomization
In 10 iterations, average +/- standard deviation
RMSE: 28.265 (+/-0.812)
MAE: 20.525 (+/-0.803)
R^2: 0.483 (+/-0.030)
60.02 seconds 

SVR
C:  RMSE, MAE, R^2 = 6.327, 3.921, 0.974
CV: RMSE, MAE, R^2 = 24.500, 15.481, 0.612
P:  RMSE, MAE, R^2 = 43.557, 43.197, 0.000

y-randomization
In 10 iterations, average +/- standard deviation
RMSE: 30.822 (+/-0.733)
MAE: 17.953 (+/-0.716)
R^2: 0.386 (+/-0.029)
342.37 seconds 

※DCVは省略
※いつまでもtest2_Tc...というネーミングではどうかと思うので、test2_を省くことにした

[todo]->[done]
モジュール化
https://twitter.com/yamasaKit_/status/1048707118885294080
GridSearchみたいにモデルとスコア指定するとシャッフルした予測制度まで返す
関数あるいはクラスとかはどうですかね？
# スコア指定はしていない また、クラス分類はやっていない



[1b1] サイトで勉強2 note.nkmk.me
Python関連記事まとめ - note.nkmk.me
https://note.nkmk.me/python-post-summary/
基礎
    基本的なエラー一覧とその原因の確認方法
    if __name__ == '__main__'の意味と使い方
    複数の変数に複数の値または同じ値を代入
    型を取得・判定するtype関数, isinstance関数
    len関数で様々な型のオブジェクトのサイズを取得
    print関数で文字列、数値および変数の値を出力
    pprintでリストや辞書を整形して出力
    無名関数（ラムダ式、lambda）の使い方
    カレントディレクトリを取得、変更（移動）
    実行ファイルの場所（パス）を取得する__file__

https://note.nkmk.me/python-if-name-main/
他のファイルからインポートされた場合は__name__変数に'<モジュール名>'が格納され、
コマンドラインからpythonコマンドで実行された場合は__name__変数に'__main__'という文字列が格納される。

したがって、if __name__ == '__main__'は
「該当のスクリプトファイルがコマンドラインから実行された場合にのみ以降の処理を実行する」という意味となる。

if __name__ == '__main__'の使い方、使いどころ
1.モジュールのテストコードを記述
2.モジュールをコマンドとして利用

PythonではC言語のようにmain()関数から処理が実行されるというわけではなく、
mainという名前の関数（main()）を定義したからといって自動的にその関数から処理が始まるわけではない。

特に大規模なプログラムの場合は慣例として起点となる関数をmain()という名前にしておくことが多いが、
これはあくまでも分かりやすさのためで、仕様として特別な意味はないし、必須でもない。
# 実際、scikit-learnではmainは使われていない


Pythonで複数の変数に複数の値または同じ値を代入
https://note.nkmk.me/python-multi-variables-values/
a, b = 100, 200
a, *b = 100, 200, 300
a = b = 100
a = b = c = 'string'

Pythonのprint関数で文字列、数値および変数の値を出力
https://note.nkmk.me/python-print-basic/
文字列の途中に変数の値を挿入して出力したい場合は以下の三つの方法がある。
    パーセント%を使うprintf形式
    文字列メソッドformat()
    f文字列（フォーマット文字列）
%d, %f.2のような変換指定子を使うC言語などのprintf形式に慣れ親しんでいるのでなければ、
公式ドキュメントにあるようにformat()メソッドやf文字列を使うのがオススメ
数値をフォーマットして出力（桁数指定など）
    print('{0:.4f} is {0:.2%}'.format(number))
さらに詳しい解説
https://note.nkmk.me/python-format-zero-hex/


Pythonのpprintの使い方（リストや辞書を整形して出力）
https://note.nkmk.me/python-pprint-pretty-print/
Pythonの標準ライブラリであるpprintモジュールを使うと、
リスト（list型）や辞書（dict型）などのオブジェクトをきれいに整形して出力・表示したり、
文字列（str型オブジェクト）に変換したりすることができる。
pprintは「pretty-print」の略。

https://note.nkmk.me/python-lambda-usage/
lambda式の具体的な使い方・活用例
sorted(), sort(), max(), min()の引数keyに使う
l_sorted_len = sorted(l, key=len)
    文字数が少ない順にソート
l_sorted_second = sorted(l, key=lambda x: x[1])
    2文字目を取得するラムダ式を引数keyに指定すると、2文字目のアルファベット順にソートされる。

