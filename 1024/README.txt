# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 10:04:42 2018

@author: Akitaka
"""
[1b] サイトで勉強2 note.nkmk.me
Python関連記事まとめ - note.nkmk.me
https://note.nkmk.me/python-post-summary/
リスト
その他
    リストを任意の値・要素数で初期化
    リストとタプルを相互に変換するlist(), tuple()
    文字列のリストと数値のリストを相互に変換
    2次元配列（リストのリスト）の行と列を入れ替える（転置）
    リストの各要素の出現個数をカウントするCounter
    複数のリストの直積（デカルト積）を生成するitertools.product
    リスト、NumPy配列、pandas.DataFrameを正規化・標準化
    Python3のmapはリストではなくイテレータを返す

以上のページのサンプルプログラムを写経完了


[1c] 書籍(の英語サイト)で勉強 Python Data Science Handbook
https://jakevdp.github.io/PythonDataScienceHandbook/

Operating on Data in Pandas
https://jakevdp.github.io/PythonDataScienceHandbook/03.03-operations-in-pandas.html

Handling Missing Data
https://jakevdp.github.io/PythonDataScienceHandbook/03.04-missing-values.html

Hierarchical Indexing
https://jakevdp.github.io/PythonDataScienceHandbook/03.05-hierarchical-indexing.html

[1c2] 欠損値の扱い
CSVファイルで欠損値をどうする？　λやωlogがないことは多々ある
空白？ Nan? None?

https://pythondatascience.plavox.info/pandas/データフレームを出力する
これを参考に、NaNを含んだデータフレームをto_csvで出力してみた
csvファイルでは欠損値NaNの部分は空白だった。
よって、「空白にする」

メモ
空白やNaNやnan　→　欠損値のNaN(dtype:float64)として扱う
Noneやnp.nan　→　文字列(dtype:object)として扱う


[1d] データベースの更新
ref:20180927,1023

[1d1] 上記の通り、欠損値を空白に修正する

[1d2]
[todo]->[done] 
ref:20180927
ScHxのref27-29を見る
https://pubs.acs.org/doi/10.1021/acs.jpcc.7b12124

検索は完了したので、更新
ref27
DOI: 10.1103/PhysRevB.96.144108

ref28
DOI: 10.1103/PhysRevB.96.094513

ref29
DOI: 10.1103/PhysRevLett.119.107001

