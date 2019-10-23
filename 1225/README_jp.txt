# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 15:24:58 2018

@author: Akitaka
"""

[1b] データベース更新
UH10
https://doi.org/10.1016/j.physleta.2018.11.048
詳細は後述



[1c] ベイズ最適化
[todo] → [done] 以下のまとめを読む
20180525 リンク集
20180528 分かりやすい説明スライドシェア
20180612 CrySPYインストール失敗　と　手法の理解
20180615 簡単なベイズ最適化テスト　と　手法の理解
20180926 scikit-learnでベイズ最適化のためのリンク　実行はなし
https://datachemeng.com/bayesianoptimization/

[1c1] 20180525から少し改良
ベイズ最適化(Bayesian Optimization, BO)
～実験計画法で使ったり、ハイパーパラメータを最適化したり～
https://datachemeng.com/bayesianoptimization/
ref:05/17
ガウス過程による回帰により説明変数と目的変数との間で回帰モデルを構築する
推定値とその分散を利用して、目的変数の値がより大きくなる(小さくなる)
可能性の高いサンプル候補を見つける
実験計画法やハイパーパラメータの最適化に活用できる
----
ベイズ最適化をするときの前提
    1.説明変数と目的変数のあるサンプルがいくつか存在する
    2.目的変数の値がなるべく大きくなるような説明変数の値の候補を知りたい
        小さくなるような～の場合は、目的変数に-1をかければいい

回帰モデルを用いた探索
= 説明変数に対して、グリッドサーチやランダムサーチ & 目的変数が最大になる説明変数を候補にする
    ただし、既存のサンプルに近い所が候補として選ばれやすい
    「候補選び→実際に実験して測定→」のループで、同じような候補ばかりになる→面白みがない
    # 私がやったhigh-Tc探索はグリッドサーチ
    # 面白味がない　→　ベイズ最適化なら違う、かのように言ってはいるが、違うのでは？
    # 後述のスライドシェアでは、逐次最適化を自動化するのが主な目的になっている

ベイズ最適化を用いた探索
=　獲得関数が最大になる説明変数を候補にする
　　目的変数の推定値だけでなく、推定値のばらつきも利用した関数
　　方法はPI,EIなど複数。

Probability of Improvement: PI(x)
= あるサンプル候補xにたいして、ガウス過程回帰を行う。
    「その推定値m(x)と分散σ^2(x)を、平均と分散とする正規分布」を、
    [既存のサンプルにおける目的変数の最大値y_max, ∞]で積分した値。
= xの推定値が、y_maxより大きくなる確率

Expceted Improvement: EI(x)
= xの推定値が、y_maxよりどれだけ大きいか(更新幅)の期待値

Mutual Information: MI(x)
= xの推定値+バラつき(実験回数ごとに更新)

適応的な実験計画法
    1.PIまたはEIの値が最大となるサンプル候補で実験する
    2.実験したら、そのサンプルを既存のサンプルに追加し、再度ガウス過程による回帰を行う
    3.1.2.を繰り返すことで、目的変数の値を大きくすることを目指す

サンプルの候補が多い時は？
1.グリッドサーチ
2.ランダムサーチ
3.遺伝アルゴリズム

サンプルプログラム
https://github.com/hkaneko1985/design_of_experiments
    demo_of_adaptive_design_of_experiments.py
    demo_of_bayesian_optimization.py
    design_of_experiments.py
全部走らせた。特に問題なし。


[1c2] 20180528から少し改良
機械学習のためのベイズ最適化入門<div>
https://www.slideshare.net/hoxo_m/ss-77421091
各項目と、次の項目の繋がりが明確！
ベイズ最適化とは？概略
→何に使うか？ハイパーパラメーター探索　職人芸
→もと簡単にできないか？グリッドサーチ
→効率的に探索できないか？ベイズ最適化
→どうやって次の観測点を判断している？獲得関数
→どうやって計算する？ガウス過程
→実行ツール

5.獲得関数(p33-39)
PI
    Probability of Improvement(改善確率)
    現在の最大値y_bestを超える確率が最も高い点を次に観測
    シンプルで分かり易いが、局所解に陥ることも

EI
    Expected Improvement(期待改善量)
    評価値と最大値の差y-y_bestの期待値が最も大きくなる点を次に観測
    PIは改善確率のみ。しかし、改善量が小さいと非効率なので、期待値を見る
    最も一般的に使われている

UCB
    Upper Confidence Bound(上側信頼限界)
    評価値の信頼区間の上限が最も高い点を次に観測
    最適解に辿り着く理論的保証がある
    ※正確な話は後述

6.ガウス過程(p40-48)
最適化したい関数がガウス過程に従うと仮定する
データから未知の関数f(x)の概形を予想
→
未観測点の期待値μと分散σ^2を算出可能
    μが大きい：周囲の観測点が大きい
    σが大きい：周囲が観測されていない
μが大きい点を次の観測点に選べば、大きい値が期待できる
しかし、そればかりでは局所解に陥る
適度にσの大きい点を探索する必要あり
→
ガウス過程を仮定することで獲得関数が計算可能
    カーネル関数(=観測点同士がどれくらい影響し合うか)を選択

おまけ(p55-67)
ランダムサーチ
    一部の機械学習手法において、ハイパーパラメータ探索に有効
    精度を左右するハイパーパラメータは少数だから
グリッドサーチ：10*10*10...としても、1つのパラメータは10点しか計算しない
ランダムサーチ：すべて異なる点を計算する

獲得関数MI
    Mutual Information
    相互情報量の増加が大きい点を次に選ぶ

アルゴリズムの評価
Regret(後悔)
    探索点におけるf(xt)と最適値f(x*)の差
    ※真の最適解と、探索点の差？
    累積Regretが小さいと良いアルゴリズム

△ UCBは最適値へ収束する理論的保証あり
◯ 正確には、「累積Regret R_T が Regret上限 G(T) √α 以下になる確率」が大きい
    Pr[R_T =< G(T) √α] >= 1-δ
    α = log(2/δ)
</div>

scikit-learnの獲得関数は？
→そもそもベイズ最適化はないので、自分で組むしかない
https://qiita.com/mokemokechicken/items/52fcb7d5057e9a5d85c4
https://thuijskens.github.io/2016/12/29/bayesian-optimisation/
サンプルコードあり

Scikit-Optimizeというのがあるらしい。これなら簡単に設定可能
http://www.kumilog.net/entry/bayesian-optimization
https://scikit-optimize.github.io/


[1c3]　20180612より少し改良
CrySPYインストール失敗　※詳細は省略

"Crystal structure prediction accelerated by Bayesian optimization"
T. Yamashita, N. Sato, H. Kino, T. Miyake, K. Tsuda, and T. Oguchi,
Phys. Rev. Materials 2, 013803 (2017).
https://doi.org/10.1103/PhysRevMaterials.2.013803

Abstract:
純粋なランダム探索と比べて、ランダム＋ベイズ最適化は、
グローバルミニマムを見つけるまでに必要な試行回数が3-4割減る
？遺伝アルゴリズムと比べて速いのかは不明

獲得関数は？
=Thompson sampling
以前(05/28)調べたPI,EI,UCBのどれでもない！

[1d] ベイズ最適化の調査
応用はハイパーパラメータ探索が多い？

ベイズ最適化入門
https://qiita.com/masasora/items/cc2f10cb79f8c0a6bbaa
    ※リンク多数
    ベイズ最適化自身はかなり昔からある
    ハイパーパラメーターサーチのためベイズ最適化がまたリバイバルしてきたらしい

機械学習のハイパーパラメータ探索: ベイズ最適化の活用
http://www.techscore.com/blog/2016/12/20/機械学習のハイパーパラメータ探索-ベイズ最適/
    探索するパラメータ次元数・取りうる範囲が増えるほど、グリッド探索での計算量は膨大になる
    →ベイズ最適化であればOK
    ただし、ベイズ最適化にもハイパーパラメータはある！

Taking the Human Out of the Loop -ベイズ最適化のすゝめ-
http://mathetake.hatenablog.com/entry/2016/12/19/145311
    ※リンク多数


[1c4]　20180615 より抜粋
[1c] テスト計算　ベイズ最適化　その１
機械学習のハイパーパラメータ探索 : ベイズ最適化の活用
http://www.techscore.com/blog/2016/12/20/機械学習のハイパーパラメータ探索-ベイズ最適/

[1d] ベイズ最適化とハイパーパラメータ探索
※20180612の延長

[1e] ベイズ最適化を行うライブラリ
ベイズ最適化でパラメータチューニングを行う
http://www.kumilog.net/entry/bayesian-optimization
https://github.com/xkumiyu/bayesian-opt
> Pythonでベイズ最適化を行うには、Scikit-Optimize (skopt)やGpyOptなどのライブラリがあります。

https://www.slideshare.net/hoxo_m/ss-77421091 のpage. 50に多数記載
例：bayesian-optimization (python)
https://github.com/fmfn/BayesianOptimization
他、COMBOも可能らしい

[1f] テスト計算　ベイズ最適化　その２
ベイズ最適化入門
https://qiita.com/masasora/items/cc2f10cb79f8c0a6bbaa
http://mathetake.hatenablog.com/entry/2016/12/19/145311


[1c5]　20180926 から抜粋
パラメータチューニング
sk-learn: ベイズ的最適化、グリッドサーチ、ランダムサーチも可能
    Pythonでベイズ最適化を使ってハイパーパラメータを探索するライブラリ実装のメモ - Qiita
    https://qiita.com/mokemokechicken/items/52fcb7d5057e9a5d85c4
    # 私見
    もしかしたら、使うかもしれないので記憶の片隅にでもとどめておく
※「[1c2] 20180528から少し改良」で既出


[1c6] サンプルプログラムまとめ
https://github.com/Ma-sa-ue/practice/blob/master/machine%20learning(python)/
    bayeisan_optimization.ipynb
    ref:
    https://qiita.com/masasora/items/cc2f10cb79f8c0a6bbaa

http://www.techscore.com/blog/2016/12/20/機械学習のハイパーパラメータ探索-ベイズ最適/
    実行した　以前は上手く行かなかったが、今回は大丈夫
    これならぎりぎり理解できる
https://thuijskens.github.io/2016/12/29/bayesian-optimisation/
    実行した？　上手く走らないけど、調べる気はしない
https://qiita.com/mokemokechicken/items/52fcb7d5057e9a5d85c4
    実行した　一応走ったものの、警告文が多い　原因を調べる気はしない　理解はしていない
https://github.com/hkaneko1985/design_of_experiments/tree/master/Python
    実行済み　ベイズ最適化のモジュールの理解はできた。後述の1-3のみを実行している。
    デモ計算はイマイチ分からんというか、１ステップしか進めてないのはなぜなのか

[1c7] 私見
ベイズ最適化の流れ
1.既知のデータからガウス過程回帰モデルを作成
2.作成した1.のモデルを用いて、未知データxに対する推定値の平均μと分散σ(事後分布)を得る。
3.2.のμとσから獲得関数(PI, EI, UCB, MIなど)を計算し、それがMaxとなる点x'を観測候補として選ぶ。
4.候補x'に対して観測を行い、結果を既知データに加える。
5.1-4を、収束するまで繰り返す。

ベイズ最適化で設定する項目
    グリッド範囲
    獲得関数
    カーネル関数（およびそのハイパーパラメータ）

昔に比べて理解しやすくなった
おそらくは、ベースであるガウス過程回帰への理解が進んだためであろう

[1c8] 余談
リッジ回帰 + カーネルトリック = ガウス過程回帰
ref:
https://www.slideshare.net/chikainoshita/8-66166352
https://qiita.com/masasora/items/cc2f10cb79f8c0a6bbaa


[todo] Tcデータで可視化してみる　※意味はよく分からないけど
0724


[3] Google Scholar おすすめの論文
"Theoretical study on UH4, UH8 and UH10 at high pressure"
Dong Wang, et al.,
Physics Letters A
    Available online 17 December 2018
    In Press, Uncorrected ProofWhat are Uncorrected Proof articles
https://doi.org/10.1016/j.physleta.2018.11.048
Highlights & Abstract
    Genetic Algorithm
    U–H system at 0-550GPa
    The UH4 is stable at 100–550 GPa.
    The Fm-3m - UH8 is most stable stoichiometry at 100–550 GPa.
    The Fm-3m - UH10 becomes stable above 450 GPa.
    The Fm-3m - UH10 is found to be a superconductor at high pressure.
    Tc = 51, 21, 12, 10 and 15 K at 100, 200, 300, 400 and 550 GPa

Figure 2.
     Predicted pressure-composition phase diagram of U–H hydrides at 0–550 GPa

Table 2. 
    Calculated ωlog, N(Ef), λ, and Tc of the R3m - UH4 and Fm-3m - UH10
    with different Coulomb pseudopotential μ⁎ of 0.1 and 0.15.
    # captionではUH4とあるが、Tableの中ではUH10のみ
    # また、本文中、UH4は超伝導にならないが、UH10は超伝導になる、と明言している。
    # > The structures of Cmca-UH4, ...
    # よって、この表に載っているのはすべて、Fm-3m - UH10 のTcであると判断する
formula,group,  P [GPa],  lambda,  w_log [K],   mu,     Tc [K]
   UH10,  225,      100,    1.05,     679.27, 0.10,      50.58
   UH10,  225,      100,    1.05,     679.27, 0.15,      39.42
   UH10,  225,      200,    0.66,     720.50, 0.10,      21.22
   UH10,  225,      200,    0.66,     720.50, 0.15,      12.51
   UH10,  225,      300,    0.54,     716.58, 0.10,      11.93
   UH10,  225,      300,    0.54,     716.58, 0.15,       5.57
   UH10,  225,      400,    0.50,     764.61, 0.10,       9.50
   UH10,  225,      400,    0.50,     764.61, 0.15,       3.85
   UH10,  225,      500,    0.53,     969.50, 0.10,      15.11
   UH10,  225,      500,    0.53,     969.50, 0.15,       6.84
-> データベースに追加
ただし、安定になるのは450GPa以上のみ
