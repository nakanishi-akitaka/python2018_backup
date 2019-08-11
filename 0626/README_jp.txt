61.3 kg+6 20:30 B  11:30+Q
6:00, 11:45, 14:15, 16:40, 20:30, 24:00 
35200 (85000) 139枚
100/35100 (85000) 139枚

活動記録

[1] 機械学習について勉強＆計算

　Numpy, PyCharmのテスト
　CuAlO2系の金属化判定プログラム＋データベース更新

　金属か絶縁体かの分類 21種類の手法を適用

　評価基準を見直す（精度→偽陽性率＆真陽性率）

11年前<div>
ノート上部
風雅は身と共に終わるべし

2007/11/26(月)
ヨーロッパ退屈日記
医療産業都市構想　スパコン＠ポーアイは生物系の知識必要？
pccluster1原因不明のトラブル
</div>


10年前<div>
ノート上部
集中と選択

2008/06/26(木)
scfとvcrelaxのストレスの差はなぜ？

V, Ta, Snの計算開始
シンプルでTcが数Kあるから
V  bcc Tc 5.3 K
Ta bcc Tc 4.5 K
Sn dia Tc 3.5 K

大学院　物性物理２　強相関電子系　
Tc計算例　詳細がある程度載っている
</div>


1年前<div>
ノート上部や日誌
26 死んで覚える
</div>



[1] 機械学習
[1a] サイトで勉強１　金子研究室
最新の投稿：なし
全ての記事は05/19~06/04で一通り読んだ
さらに06/05~06/25でもう一週した

今度は、以下のカテゴリーを順次読む
データ解析に関するいろいろな手法・考え方のまとめ
https://datachemeng.com/summarydataanalysis/

数学(行列計算・線形代数・統計・確率)が不安な方へ
1.高校数学の知識から、人工知能・機械学習・データ解析へつなげる、必要最低限の教科書
2.人の学習と人工知能の学習～人工知能を学習させるとかモデルを構築するということ～


[1b] サイトで勉強２　Python でデータサイエンス
https://pythondatascience.plavox.info/

[1b1] Python の開発環境
Python のインストール→
Anaconda を利用した Python のインストール (Windows)
https://pythondatascience.plavox.info/pythonのインストール/pythonのインストール-windows

ここが初耳
> Anaconda をインストールした後は、PyCharm のインストール のページから、
> Python の統合開発環境、PyCharm をインストールしましょう。

Anaconda を利用した Python パッケージのインストール
https://pythondatascience.plavox.info/pythonのインストール/anacondaを利用したpythonパッケージのインストール
Anaconda Promptを起動して、
$ pip install [package name]


Python の開発環境
https://pythondatascience.plavox.info/Python の開発環境
<div>
本サイトでは、データ分析の用途で用いられる主要な IDE として、
本サイトでは IDE の「PyCharm」と「Rodeo」、
ノートブックの「Jupyter Notebook」について、
インストール方法、使い方を以下ページにて紹介しています。

IDE とノートブックにはそれぞれ、向き、不向きがあります。

例えば、企業の本番環境で稼働させるバッチアプリケーションを数人のチームで開発する場合は、
PEP8 のようなコーディング規則の統一機能や Git のようなレポジトリとの連携機能が備わっている 
IDE のほうが使い勝手が良いでしょう。

逆に個別のアドホック型の分析や学習目的での利用としては、
過去に試した設定値やロジックを振り返って確認できるため、
ノートブック型のほうが比較的適していると言えます。

どちらを使って作業するかについて迷った場合は、自分自身にとって使い勝手が良く、
慣れている環境を使って作業するのがオススメです。
慣れた環境での作業と不慣れな環境では、場合によっては、
生産性に数倍から数十倍の差が生まれることもあります。
</div>


https://pythondatascience.plavox.info/pythonの開発環境/pycharmのインストール
https://pythondatascience.plavox.info/pythonの開発環境/pycharmを使ってみる
https://pythondatascience.plavox.info/pythonの開発環境/pycharmの便利な使い方
矩形選択、予測入力、変数一括置換、diff
PEP8に従うよう、自動整形
https://twitter.com/hirokaneko226/status/1001762797091368960

https://pythondatascience.plavox.info/pythonの開発環境/jupyter-notebookを使ってみよう
test1.ipynbでマークダウンのテスト
使い方
1.Codeの部分をクリックしてMarkdownを選択
2.Markdownを書き込みたいセルにMarkdownで記入
3.Ctrl+Enterで記入したセルを実行
4.Markdownが変換される
5.編集したい場合は、セルをダブルクリック

https://pythondatascience.plavox.info/pythonの開発環境/rodeo のインストールと使い方
IDE としての基本的な機能は、PyCharm などと同じですが、
RStudio に使い慣れている方には、Rodeo は扱いやすいのではないかと思います。



[1b2] NumPy で行列計算
https://pythondatascience.plavox.info/numpy
NumPy で行列を作ってみよう
  test2.py
NumPy で行列を操作する
  test2.py
NumPy で数学系の関数を使ってみよう
  test2.py
NumPy で線形代数
  test3.py
Numpy で乱数を生成する
  test4.py
NumPy で金融・財務系の計算を試してみよう
  test5.py


[1c] PyCharmインストール
C:\Program Files\JetBrains\PyCharm Community Edition 2018.1.4
してみたけど、使いづらい。
何をどうすればいいのか分かりづらい
hello worldすらすぐには出来なかった
デフォルトの敗色が暗くて見づらい
→
アンインストール

結局、python使うにはどれがいい？
軽く検索した範囲では、とりあえずSpyderがよさそう


[1d] ギャップを機械学習してみる　分類編
Egの数値を「回帰」ではなく、金属か絶縁体かを「分類」する
ref:06/22, 25

[1d1] 少しアップデート
test1.csv: Eg の数値
test1_cls.csv: 絶縁体なら2, 金属なら1
とわざわざ2種類用意していた
ステップ関数：y = 1 * (x > 0) 
　を用いることでデータベースは１つだけで済むようになった
　※絶縁体なら１　金属なら０に変更
一々データベースをコピーしておくのもかさばるので
　更新がない限り、別のディレクトリにあるものを参照する方式に変更
test6.py でテスト
test7.py で本体更新


[1d2] フォルダ変更
Download\python\201806xx 
Download\python\06xx 

[1d3] 精度評価の見直し
ref:25
> 全部絶縁体に分類したものもある。それでも元々絶縁体が多いと精度が上がってしまう！
> →AUC、ROC曲線を使うべきでは ref:04/10-11

0410, 0411より<div>
【ROC曲線とAUC】機械学習の評価指標についての基礎講座
http://www.randpy.tokyo/entry/roc_auc
単純に正解率で判断すると、クラスの偏りに左右されてしまう

偏りに影響されない評価指標
偽陽性率：正解データが負を間違って正と予測した割合
真陽性率：正解データが正を正しく正と予測した割合
ROC曲線：偽陽性率を横軸に、真陽性率を縦軸に置いてプロットしたもの
AUC：ROC曲線の面積
</div>


方法
http://scikit-learn.org/stable/modules/model_evaluation.html
これを参考に、偽陽性率、真陽性率を計算してみた
test8.py　偽陽性率、真陽性率のテスト計算　
test7.py　本体のアップデート

結果
Accuracy, False Positive Rate, True Positive Rate
0.719, 0.962, 0.971, Linear Discriminant Analysis
0.729, 1.000, 1.000, Linear Support Vector Machine
0.854, 0.538, 1.000, Non-Linear Support Vector Machine
0.760, 0.731, 0.943, Quadratic Discriminant Analysis
0.781, 0.731, 0.971, k-Nearest Neighbor Classification
0.740, 0.731, 0.914, Gaussian Naive Bayes
1.000, 0.000, 1.000, Decision Tree
0.958, 0.077, 0.971, Random Forests
0.729, 1.000, 1.000, Gaussian Process Classification
0.729, 1.000, 1.000, Bagging[LDA]
0.729, 1.000, 1.000, Bagging[LSVM]
0.875, 0.423, 0.986, Bagging[NLSVM]
0.740, 0.731, 0.914, Bagging[QDA]
0.760, 0.654, 0.914, Bagging[kNNC]
0.719, 0.846, 0.929, Bagging[NB]
0.969, 0.038, 0.971, Bagging[DT]
1.000, 0.000, 1.000, Bagging[GPC]
0.729, 1.000, 1.000, AdaBoost[LSVM]
0.729, 1.000, 1.000, AdaBoost[NLSVM]
0.740, 0.846, 0.957, AdaBoost[NB]
1.000, 0.000, 1.000, AdaBoost[DT]

http://www.randpy.tokyo/entry/roc_auc
> 良いモデルとは偽陽性率が低い時点で既に真陽性率が高い数値がでること
故に、
FPRが低い＆TPRが高い：DR、RF、Bagging[DT, GPC], AdaBoost[DT] が優秀
