# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 15:26:30 2018

@author: Akitaka
"""

[1a3] ソフトセンサー
プロセス制御・プロセス管理・ソフトセンサーの概要と研究の方向性 (化学プラントにおけるデータベース利用)
https://datachemeng.com/processcontrolsoftsensor/
化学プラント・産業プラント
    温度・圧力・流量・濃度といった いろいろなプロセス変数を制御しながら運転する必要がある
うまく制御・管理できないケースとは
    1.たくさんのプロセス変数をまとめて管理する (多変量プロセス管理)
        正常なデータが存在する適切なデータ領域を決める
        異常と診断されたときにどのプロセス変数が異常に関わっているか診断する
        異常の原因を解明する
    2.プロセス変数の値がすぐに測定できない
    -> 推定しながら制御する
    ソフトセンサー
        測定が難しいプロセス変数の値を、推定する方法
        過去に測定されたデータを用いて、温度・圧力などの簡単に測定できるプロセス変数と、
        濃度・密度などの測定が難しいプロセス変数との間で作成した、回帰モデルのこと
ref
時系列データを扱うときの３つの注意点(ソフトセンサー解析など)[データ解析用のPythonプログラム付き]
https://datachemeng.com/pointsoftimeseriesdataanalysis/
適応型ソフトセンサーで産業プラントにおけるプロセス状態等の変化に対応する(Adaptive Soft Sensor)
https://datachemeng.com/adaptivesoftsensors/




[1a4] クリギング
クリギング (Kriging) の仕組み
http://desktop.arcgis.com/ja/arcmap/10.3/tools/3d-analyst-toolbox/
    how-kriging-works.htm

https://en.wikipedia.org/wiki/Kriging
    In statistics, originally in geostatistics,
    kriging or Gaussian process regression is a method of interpolation
    for which the interpolated values are modeled
    by a Gaussian process governed by prior covariances.
ガウス過程回帰と同様？　内挿らしい？
Wikipediaの画像は、ガウス過程回帰のに似ている

http://nobunaga.hatenablog.jp/entry/2015/09/15/221358
Krigingもガウス過程も含めて，ノンパラメトリック回帰は近傍との滑らかな関係性を仮定している．

https://jp.mathworks.com/help/stats/gaussian-process-regression.html
ガウス過程回帰モデル (クリギング)
# イコール扱い？

https://jp.mathworks.com/help/stats/gaussian-process-regression-models.html
以下のクリギングの数式(p.5)と同様の数式が出てくる！やっぱり同じ？
    http://www.gisa-japan.org/dl/19-2PDF/19-2-59.pdf

http://www.kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/1584-12.pdf
Krigingの式の中に、ガウス過程回帰が出てくる
2008年のなので、やや古い

ref:
https://www.jstage.jst.go.jp/article/tjpnsec/3/3/3_173/_pdf/-char/ja

ガウス過程回帰の勉強に
https://www.ism.ac.jp/~daichi/lectures/H26-GaussianProcess/
    gp-lecture1-matsui.pdf
https://www.slideshare.net/KeisukeSugawara/slide0629
    カーネル関数の最適化
http://machine-learning.hatenablog.com/entry/2018/01/13/142612
    深層学習をガウス過程で構築しようとする論文について
https://www.yasuhisay.info/entry/20091011/1255189429
    ガウス過程のメリットデメリット
    


[1b] Tcデータベースの可視化 by PCA
[todo]->[done] Tcデータで可視化してみる　※意味はよく分からないけど

主成分の寄与率
[0.43470527 0.21100239 0.17253513 0.09774241 0.0840148 ]

5つ中1つは、水素の原子番号で固定なのでほぼ無意味
もう一つ、非水素原子の個数はだいたい1個なので、ほぼ無意味なのでは？

ref:
https://scikit-learn.org/stable/modules/generated/
    sklearn.decomposition.PCA.html
https://blog.amedama.jp/entry/2017/04/02/130530
http://neuro-educator.com/ml21/
https://www.haya-programming.com/entry/2018/03/27/024144
http://inaz2.hatenablog.com/entry/2017/01/23/214409
pandasデータフレームの列の入れ替え【Python3】
https://to-kei.net/python/data-analysis/change_columns/



[1c] ベイズ最適化 + kNN
[1b] ベイズ最適化について補足
[?] ベイズ最適化はガウス過程回帰でのみ可能？
    k最近傍法やアンサンブル学習でも平均＋標準偏差を計算できる。それらを使うのは？
    ベイズ確率とは別物なので、ややずれた話かもしれないが
    あるいは、混合ガウスモデル回帰は？

[todo]->[done] kNNで、y_pred = μ だけでなく、σも活用して、ベイズ最適化もどきをやってみる
    よく考えたらマズイのでは？
    単純にやってしまうと、μもσも同じになる未知サンプル = 次の候補が多数でてしまう。
     -> 既知サンプルとの距離も考慮に入れれば回避できる

ということで、σ に ad_knn を応用して、ベイズ最適化モドキをやってみた

結果
ガウス過程回帰
    最初は2点でいい
    15回もあれば収束
k最近傍法
    最初はk点以上必要 ※設定次第だが
    30-40回ぐらいで収束 ※k=2として、最初の点もガウス過程回帰同様、2点とした場合

結論
ガウス過程回帰でなくても可能。
    収束は遅い。
    ハイパーパラメータを予め決めなければならない。
σをデータ密度で近似するのであれば、kNN以外のものでも応用は可能。
