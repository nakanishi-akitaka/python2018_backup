# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 11:23:32 2018

@author: Akitaka
"""

[1a2]
ナイーブベイズについて
https://datachemeng.com/naivebayesclassifier/

https://ja.wikipedia.org/wiki/単純ベイズ分類器
議論
> 独立性の仮定を広範囲に適用することが正確性に欠けるという事実があるにもかかわらず、
> 単純ベイズ分類器は実際には驚くほど有効である。特に、クラスの条件付き特徴分布を分離することは、
> 各分布を1次元の分布として見積もることができることを意味している。
> そのため、特徴数が増えることで指数関数的に必要なデータ集合が大きくなるという
> 「次元の呪い」から生じる問題を緩和できる。MAP 規則を使った確率的分類器の常として、
> 正しいクラスが他のクラスより尤もらしい場合に限り、正しいクラスに到達する。
> それゆえ、クラス確率はうまく見積もられていなくてもよい。
> 言い換えれば、根底にある単純な確率モデルの重大な欠陥を無効にするほど、
> 分類器は全体として十分に頑健である。
> 単純ベイズ分類器がうまく機能する理由についての議論は、後述の参考文献にもある。 
つまり、確率の「数値」はいい加減であっても、「大小関係」さえ合っていれば、分類は正しく行える、
のがうまくいく原因であるらしい。

ナイーブベイズ分類器を頑張って丁寧に解説してみる
https://qiita.com/aflc/items/13fe52243c35d3b678b0

ナイーブベイズについて勉強したのでざっくりまとめ
https://dev.classmethod.jp/machine-learning/naivebaise_zakkurimatome_/
> ただ、文書中の「単語間の関係」についての仮定があまりにもシンプルすぎるが故に、「精度が低い」と指摘されているのも事実です。
> いざ、文書分類をするためのアルゴリズムは多々あるので、
> いざ実務で使う際は色々なアルゴリズムの特徴を把握する必要がありますが、
> 「とりあえず文書分類をしてみて雰囲気を感じ取りたい」、
> という場合はナイーブベイズを試してみるのも有効だと思います。

機械学習アルゴリズム〜単純ベイズ法〜
http://rtokei.tech/machine-learning/機械学習アルゴリズム〜単純ベイズ法〜/
> 単純ベイズ分類器は英語ではNaive Bayes classifierと呼ばれます。
> Naive Bayes classifierを訳すると「単純」と付けられていますが、
> アルゴリズムが単純でわかりやすいという意味ではなく、
> 「Naive=うぶな、ばか正直な、考えが甘い」という「データについてNaiveな想定」を置いているために、
> 単純という言葉につながっているようです。
> 決して、アルゴリズムの精度が悪いという意味でもなければ、
> テキスト分類では、標準的なアルゴリズムとして広く利用されています。
> 
> 「データについてNaiveな想定」とは、どういった意味なのかと言いますと、
> 各特徴が独立という仮定を置いているためです。
> 基本的に各言葉は独立ではない場合が多くあります。
> 例えば、スパムメールには、「あの言葉が入っていれば、この言葉も入っている」ということが多いので、
> 言葉は独立な関係ではありません。
> にも関わらず、単純ベイズ分類器は非常に正しく分類が実行できてしまうようです。

ナイーブベイズを用いたテキスト分類
http://aidiary.hatenablog.com/entry/20100613/1276389337
> 今回取り上げるのは、よく使われていて実装も簡単、しかも高速というナイーブベイズです。
> 精度評価のベースラインとしてよく使われてます。

ゼロ頻度問題
未知の文書のカテゴリを予測する際、訓練データのボキャブラリに含まれない単語を1つでも含んでいると
単語の条件付き確率は0となり、単語の条件付き確率の積で表される確率も0となる

機械学習入門者向け Naive Bayes(単純ベイズ)アルゴリズムに触れてみる
https://avinton.com/academy/naive-bayes/
長所 	
    単純(実装も簡単)かつ強力
    とても大きなデータセットに対しても有効
    高速で計算資源も少なくてよい
    少ないトレーニングデータでも性能が出る
    重要でない特徴量の影響を受けにくい
短所 	
    各特徴量が独立であると仮定しなければならない(実データでは成り立たないことも多い)
応用先
    リアルタイムでの処理、テキスト分類

※短所について
> つまり各特徴量が独立に推定結果に影響します。
> これはとても強い仮定で、実データでは成り立たないことも多いです。
> 実はこれがNaiveと名前につく所以でもあるのですが、
> その強い仮定(制約)にもかかわらず、この仮定が成り立たないであろう実データでも、
> 驚くほどよい結果を出すというのがこのアルゴリズムの優秀な点です。
「とりあえずナイーブベイズでテストしてみよう」と考えたくなる

テキスト分類の場合、特徴量は単語の頻度になるので、確率は多項分布やベルヌーイ分布
正規分布ではない


[1b] サイトで勉強2 note.nkmk.me
Python関連記事まとめ - note.nkmk.me
https://note.nkmk.me/python-post-summary/
日付・時間
    datetimeで日付や時間と文字列を変換（strftime, strptime）
    日付から曜日や月の名前を日本語文字列で取得

URL
    URLエンコード・デコード（urllib.parse.quote, unquote）
    URLのクエリ文字列（パラメータ）を取得・作成・変更

CSV・Excel・JSON
    コンマの後に空白があるcsvを読むときは注意
    Excelファイルを扱うライブラリの比較
    Excelファイル（xlsx）を読み書きするopenpyxlの使い方
    Excelファイルを読み込み・書き込みするxlrd, xlwtの使い方
    JSONファイル・文字列の読み込み・書き込み

以上のページのサンプルプログラムを写経(コピペ)完了




[1c] 書籍(の英語サイト)で勉強 Python Data Science Handbook
https://jakevdp.github.io/PythonDataScienceHandbook/

In-Depth: Support Vector Machines
https://jakevdp.github.io/PythonDataScienceHandbook/05.07-support-vector-machines.html

In-Depth: Decision Trees and Random Forests
https://jakevdp.github.io/PythonDataScienceHandbook/05.08-random-forests.html

In Depth: Principal Component Analysis
https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
> noisy = np.random.normal(digits.data, 4)
digits.dataの各値をμ、分散を4とした乱数を、digitsのサイズに合わせて生成
4->0.001にするとほぼ同じになる

In-Depth: Manifold Learning
https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html
多用体学習：PCAのような次元削減、非線形

以上のページのサンプルプログラムを写経(コピペ)完了
