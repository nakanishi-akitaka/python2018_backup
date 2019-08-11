# -*- coding: utf-8 -*-
"""
Pandas のデータフレームを CSV ファイルやテキストファイルに出力する
このページでは、Pandas を用いて作成したデータフレームや Pandas を用いて加工したデータを 
CSV ファイルやテキストファイルとして書き出す方法 (エクスポートする方法) についてご紹介します。
Created on Wed Jun 27 14:06:33 2018

@author: Akitaka
"""
# https://pythondatascience.plavox.info/pandas/データフレームを出力する

# CSV ファイルとして出力する: DataFrame.to_csv() メソッド
# Pandas には、CSV ファイルとして出力するメソッドとして、DataFrame.to_csv() メソッドが存在します。
# また、この際、区切り文字を CSV ファイルで用いるカンマ (,) から タブ (\t) などへ置き換えることで、
# テキストファイルとして出力する事もできます。

# コード例
# 以下に実際に作成した Pandas のデータフレームをファイルとして出力するコードの例を紹介します。

# CSV ファイルとして出力する
import pandas as pd
 
# データフレームを作成
df = pd.DataFrame([
  ["0001", "John", "Engineer"],
  ["0002", "Lily", "Sales"]],
  columns=['id', 'name', 'job'])
 
# CSV ファイル (employee.csv) として出力
df.to_csv("employee.csv")


# テキストファイルとして出力する

# テキストファイル (employee.txt) として出力
df.to_csv("employee.tsv", sep="\t")
# ↑は、サイトの例が間違っていたので自分で修正した
# 参考
# https://note.nkmk.me/python-pandas-to-csv/


# コード例 (日本語を含む場合)
# 日本語の文字列を含んだデータセットを出力する場合は以下のように、encoding="<文字コード名>" 
# を引数に指定します。

# Windows 版のExcel で読み込みできる形式で出力する場合は、以下のように、
# シフト JIS 形式で出力する必要があります。
# Python で使える文字コードの一覧は 7.2.3. Standard Encodings にあります。

import pandas as pd
 
# データフレームを作成
df = pd.DataFrame([
  ["1001", "山田 裕司", "エンジニア"],
  ["1002", "佐々木 美紀", "営業"]],
  columns=['id', 'name', 'job'])
 
# Shift-JIS 形式の CSV ファイル (employee.sjis.csv) して出力
df.to_csv("employee.sjis.csv", encoding="shift_jis")