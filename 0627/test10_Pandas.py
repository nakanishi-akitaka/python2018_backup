# -*- coding: utf-8 -*-
"""
Pandas で CSV ファイルやテキストファイルを読み込む
このページでは、CSV ファイルやテキストファイル (タブ区切りファイル, TSV ファイル) を読み込んで 
Pandas のデータフレームに変換する方法について説明します。
Created on Wed Jun 27 13:42:05 2018

@author: Akitaka
"""
# https://pythondatascience.plavox.info/pandas/csvファイルの読み込み
# Pandas のファイルの読み込み関数
# CSV ファイルのロード: read_csv()
# Pandas には、CSV ファイルをロードする関数として、read_csv() メソッドが用意されています。

# テキストファイルのロード: read_table()
# テキストファイルなど、一般的な可変長のテキストファイルを読み込む関数として、read_table() メソッド
# が用意されています。

# CSV ファイル / テキストファイル の読み込み例 (ローカルファイル)
# 事前に用意したファイルを読み込むには、Pythonファイルと同じフォルダにファイルを配置し、
# ファイル名を直接指定します。

# データが手元にない場合は、以下からサンプルデータをダウンロード可能です。
# http://pythondatascience.plavox.info/wp-content/uploads/2016/05/
# sample_dataset.csv
# sample_dataset.txt

# CSV ファイルを読み込み
import pandas as pd
dataset1 = pd.read_csv("sample_dataset.csv")
print(dataset1)

# テキストファイルを読み込み
dataset2 = pd.read_table("sample_dataset.txt")
print(dataset2)

#%%
# コードの例 (日本語を含むファイルを読み込む例)
# 日本語や韓国語、中国語などのマルチバイト文字を含むファイルを読み込む場合は、
# 引数に encoding="<文字コード>" を指定することで正しく文字化けしない状態で
# 読み込むことができます。
# Python で使える文字コードの一覧は 7.2.3. Standard Encodings にあります。
import pandas as pd
 
# UTF-8 形式の CSV ファイルを読み込み
dataset3 = pd.read_csv("sample_dataset.utf8.csv", encoding="utf_8")
print(dataset3)

# Shift-JIS 形式の CSV ファイルを読み込み
dataset4 = pd.read_csv("sample_dataset.sjis.csv", encoding="shift_jis")
print(dataset4)

# EUC-JP 形式のテキストファイルを読み込み
dataset5 = pd.read_table("sample_dataset.eucjp.txt", encoding="euc_jp")
print(dataset5)

#%%
# CSV ファイル / テキストファイルの読み込み例 (URL を指定)
# インターネット上に配置されたファイルを読み込むことも可能です。
# 本例では、当サイトにアップロード済みのCSV ファイルやテキストファイルを読み込みます。
# CSV ファイルを読み込み
dataset3 = pd.read_csv("http://pythondatascience.plavox.info/wp-content/uploads/2016/05/sample_dataset.csv")
print(dataset3)

# テキストファイルを読み込み
dataset4 = pd.read_table("http://pythondatascience.plavox.info/wp-content/uploads/2016/05/sample_dataset.txt")
print(dataset4)

# 自作部分：データセット→numpy配列に変換
import numpy as np
print(np.array(dataset3))
