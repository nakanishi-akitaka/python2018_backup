# -*- coding: utf-8 -*-

[1d] 論文追試
https://arxiv.org/pdf/1803.10260.pdf
Table 1,2から、80個の説明変数を作成するテスト
  test1.py
  
[1e1] 物理量を取得
欠損値のチェック
  test2.py
mendeleevの値と、wikipediaの値と、明らかに合っていない！
単位がどちらもkJ/molのはずなのに全然合わない

[1e2] データベースの作成
  test3.py
  arxiv.1803.10260.csv
[電算]Mathematicaの無料版を見つけた
http://d.hatena.ne.jp/superstring04/20170506/1494066684
Mathics
https://mathics.angusgriffith.com/
オンラインで動かせる
例：
ElementData[1,"ElectronAffinity"]と入力して[shift]+[enter]で
水素の電子親和力72.77が出力される

ただし、これでもヘリウムやベリリウムの電子親和力はない！

ElementData["Properties"]で調べたところ、"Valence"だけはない
本家Mathematicaにはある？

以下のコマンドで、"Valence"以外は全部出力できる（原子番号10まで）
Table[{
ElementData[z, "AtomicNumber"], 
ElementData[z, "Abbreviation"],
ElementData[z, "AtomicWeight"], 
ElementData[z, "IonizationEnergies"],
ElementData[z, "AtomicRadius"], 
ElementData[z, "Density"],
ElementData[z, "ElectronAffinity"], 
ElementData[z, "FusionHeat"], 
ElementData[z, "ThermalConductivity"], 
}, {z, 10}]

最後に//TableFormを付けると見やすいが、コピペしたときに、コンマが入らなくなるのが不便
Mathematicaの出力コピペを改変し、プログラムを作成
Missing[NotApplicable], Missing[NotAvailable] -> NaN
arxiv.1803.10260.csvとして出力した
ただし、NaNとしたものが空白になっている点に注意

ref
http://mathematica-guide.blogspot.com/2012/02/element-data.html


[1e3] 回帰計算
  test1.py
とりあえずSVR＋デフォルトのハイパーパラメータ
物理量が無いものはスキップすることにした

