# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 14:08:06 2018

@author: Akitaka
"""
[1b2] docstring
Pythonのdocstring（ドキュメンテーション文字列）の書き方
https://note.nkmk.me/python-docstring/
書き方のフォーマットは特に統一されていない。
代表的な3つのスタイルの例を示す。
    reStructuredText（reST）スタイル
    NumPyスタイル
    Googleスタイル
Python3.0以降では関数アノテーション（Function Annotations）という仕組みによって、
関数の引数や返り値にアノテーション（注釈）となる式を記述することができる。
    def func_annotations_type(x: str, y: int) -> str:
        return x * y
他ref
https://qiita.com/simonritchie/items/49e0813508cad4876b5a

C:\Users\Akitaka\Downloads\python\1009\my_library.py
について、docstringを書いてみた
