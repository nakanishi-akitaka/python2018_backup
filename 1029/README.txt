# -*- coding: utf-8 -*-
"""
https://note.nkmk.me/python-f-strings/
Created on Mon Oct 29 12:27:32 2018

@author: Akitaka
"""

[1a2] 新着記事
Sparse Generative Topographic Mapping(SGTM): 
データの可視化とクラスタリングを一緒に実行する方法 [金子研論文] 
https://datachemeng.com/sgtm/
https://doi.org/10.1021/acs.jcim.8b00528
比較１
GTM：
    データの可視化手法、SOMの上位互換
SGTM:
    GTM のアルゴリズムを少し改良して、モデルを sparse にすることで、
    データの可視化だけでなくクラスタリングも一緒にできるようになった
比較２
GTM：
    混合係数(負担率)πk=1/(GTMマップのグリッド数)=1/(マップサイズ)**2、共分散を0,分散をある値に固定した GMM
    GMMを二次元に落とし込んだもの。
GMM:
    クラスタリング手法。混合係数 πk が可変。
SGTM:
    πkを可変にして、W, β と一緒に Expectation-Maximization(EM)アルゴリズムで πk を最適化する手法
    ベイズ情報量基準(Bayesian Information Criterion, BIC)を用いて、クラスター数が自動できまる。

テスト：
    QSPR のデータセットや QSAR のデータセットを解析
結果：
    データの可視化の性能を表す指標 k3n error の値もほとんど変わらない
    GTM と同様に可視化ができ、さらにクラスタリングも可能であることを確認
    各サンプルに自動的にクラスターが割り当てられ、色付きのサンプルとして二次元にプロットされて、とても見やすい

サンプルコード
https://github.com/hkaneko1985/gtm-generativetopographicmapping


[1b] サイトで勉強2 note.nkmk.me
Python関連記事まとめ - note.nkmk.me
https://note.nkmk.me/python-post-summary/
書式変換（フォーマット）
    format関数・メソッドで書式変換（0埋め、指数表記、16進数など）
    f文字列（フォーマット済み文字列リテラル）の使い方
    文字列・数値をゼロ埋め（ゼロパディング）
    文字列・数値を右寄せ、中央寄せ、左寄せ

数値と変換
    2進数、8進数、16進数の数値・文字列を相互に変換
    数字の文字列を数値に変換
    文字列が数字か英字か英数字か判定・確認

その他
    正規表現モジュールreの関数match、search、sub
    半角1文字、全角2文字として文字数（幅）カウント
    splitでカンマ区切り文字列を分割、空白を削除しリスト化
    文字列を折り返し・切り詰めして整形するtextwrap
    Unicodeエスケープされた文字列・バイト列を変換


以上のページのサンプルプログラムを写経(コピペ)完了

https://note.nkmk.me/python-f-strings/
> Python3.6からf文字列（f-strings、フォーマット文字列、フォーマット済み文字列リテラル）
> という仕組みが導入され、冗長だった文字列メソッドformat()をより簡単に書けるようになった。
裏返せば、短くかけるという程度の意味しかなさそう？
あと、分かりやいのは間違いない
違いはあった！
> 文字列メソッドformat()との違い
> f文字列では式を使用可能
> 辞書のキー指定方法


https://note.nkmk.me/python-zero-padding/
> 文字列に対してパーセント演算子を使うことで、書式変換した文字列を取得することができる。
> なお、公式ではformat()が推奨されている。