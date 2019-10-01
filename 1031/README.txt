# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:37:55 2018

@author: Akitaka
"""

[1b] サイトで勉強2 note.nkmk.me
Python関連記事まとめ - note.nkmk.me
https://note.nkmk.me/python-post-summary/
ファイル・ディレクトリ（フォルダ）
    ファイルの読み込み・書き込み
        ファイルの読み込み、書き込み（作成・追記）
        新しいディレクトリにファイルを作成・保存
        ファイル内の任意の文字列を含む行を抽出（grep的処理）
        Web上の複数の画像ファイルを一括ダウンロード・保存
    
    ディレクトリ（フォルダ）の作成
        ディレクトリ（フォルダ）を作成するmkdir, makedirs
        深い階層のディレクトリを再帰的に作成するmakedirs
    
    存在確認・サイズ取得
        ファイル、ディレクトリ（フォルダ）のサイズを取得
        ファイル、ディレクトリ（フォルダ）の存在確認
    
    パス（ファイル名・ディレクトリ名）の処理
        パス文字列からファイル名・フォルダ名・拡張子を取得、結合
        ファイル名・ディレクトリ名の一覧をリストで取得
        ファイル名の前後に文字列や連番を加えて一括変更
    
    zip圧縮・解凍
        zipファイルを圧縮・解凍するzipfile
        ディレクトリ（フォルダ）をzipに圧縮

以上のページのサンプルプログラムを写経(コピペ)完了




[1c] 書籍(の英語サイト)で勉強 Python Data Science Handbook
https://jakevdp.github.io/PythonDataScienceHandbook/

In Depth: Naive Bayes Classification
https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html

In Depth: Linear Regression
https://jakevdp.github.io/PythonDataScienceHandbook/05.06-linear-regression.html

分類/回帰の基本がそれぞれ、ナイーブベイズと線形回帰らしい
どちらも、最適化するべきハイパーパラメータが存在しない

Q.ナイーブベイズはガウス関数使うなら、パラメータはあるのでは？
A.自動的に決まるらしい
https://datachemeng.com/naivebayesclassifier/
> yのクラスごとに、説明変数Xごとに、データセットから平均μ(xi,y)標準偏差σ(xi,y)を計算

以上のページのサンプルプログラムを写経(コピペ)完了
