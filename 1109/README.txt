# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 11:09:45 2018

@author: Akitaka
"""

[1b] MNISTでテスト計算
ディープフォレストで使用していたMNISTは画像認識の入門用サンプルらしい
https://employment.en-japan.com/engineerhub/entry/2017/04/28/110000
論文を読む限り、SVRやRFでも95%以上の精度は出せる

kNNでテスト計算した場合でさえ、95%以上の精度が出る & 約30分かかる
ハイパーパラメータの最適化なし
Accuracy: 0.9717142857142858
F1 score: 0.9716759551308176
2061.73 seconds 


[1c1] 論文読み
20180612より<div>
ベイズ最適化で構造探索
T. Yamashita, N. Sato, H. Kino, T. Miyake, K. Tsuda, and T. Oguchi,
Phys. Rev. Materials 2, 013803 (2017).
https://doi.org/10.1103/PhysRevMaterials.2.013803

Title:Crystal structure prediction accelerated by Bayesian optimization
Abstract:
純粋なランダム探索と比べて、ランダム＋BOは、グローバルミニマムを見つけるまでに必要な試行回数が3-4割減る
？遺伝アルゴリズムと比べて速いのか？

獲得関数は？
=Thompson sampling
以前(05/28)調べたPI,EI,UCBのどれでもない！
</div>


[1c2] 論文読み
～を引用している論文 4本
"Accelerating atomic structure search with cluster regularization featured"
K. H. Sørensen, M. S. Jørgensen, A. Bruix, and B. Hammer
J. Chem. Phys. 148, 241734 (2018)
https://doi.org/10.1063/1.5023671 
機械学習を使った構造最適化
cryspyのことは、機械学習による構造最適化の先行研究として紹介されている
ランダム探索の欠点(局所不安定?により収束が遅い)をあげる中で出てくる


Data-Driven Learning of Total and Local Energies in Elemental Boron
Volker L. Deringer, Chris J. Pickard, and Gábor Csányi
Phys. Rev. Lett. 120 156001 (2018)
https://doi.org/10.1103/PhysRevLett.120.156001
機械学習を使った構造最適化
cryspyのことは、機械学習による構造最適化の先行研究として紹介されている
他にも機械学習による構造最適化の例が挙げられている
↓↓の、論文と同じ研究グループによるもの。計算手法も同じ
ホウ素に適用して、その性能のデモンストレーションを行っている様子


"Fine-grained optimization method for crystal structure prediction"
Kei Terayama, Tomoki Yamashita, Tamio Oguchi & Koji Tsuda 
npj Computational Materialsvolume 4, Article number: 32 (2018) 
https://doi.org/10.1038/s41524-018-0090-y
同じ小口先生
ランダムに結晶構造をたくさん作って、最適化を「途中まで」行い、
EtotやForce(から計算した評価値)が一番小さい構造を選び、それのみを「最後まで」最適化する
これにより、計算量を減らす
この手法＋cryspyを使った計算らしい


"Data-driven learning and prediction of inorganic crystal structures"
Volker L. Deringer,*ab  Davide M. Proserpio,cd  Gábor Csányia  and  Chris J. Pickard
Faraday Discuss., 2018,211, 45-59 
https://doi.org/10.1039/C8FD00034D
機械学習を使った構造最適化
サンプルとしてリンが出てくる
cryspyのことは、機械学習による構造最適化の先行研究or最近の事例として紹介されている
詳細な比較はされていない
第一原理計算により、EtotやForceのデータベースを作成し、
そこから機械学習させてEtotやForceを計算させている？


[1c3] 
https://doi.org/10.1103/PhysRevLett.120.156001
より
To overcome the performance and scaling limitations of DFT, 
machine learning (ML) is increasingly used these
days for both creating interatomic potential models [25]
and for speeding up structure-searching algorithms [26,27].
Fast ML-based potentials have been suggested to be useful
for driving structure searches[28], and indeed, we have
shown that a ML potential initially fitted for liquid and amorphous carbon [29]
can be used to discover hitherto unknown hypothetical carbon allotropes [30]. 
However, many carbon (as well as silicon) networks can readily be generated
 by direct enumeration [31–33].

[25] For a recent overview, see J. Behler, J. Chem. Phys. 145, 170901 (2016).
https://doi.org/10.1063/1.4966192

[26] T. Yamashita, N. Sato, H. Kino, T. Miyake, K. Tsuda, and T. Oguchi, Phys. Rev. Mater. 2, 013803 (2018).
https://doi.org/10.1103/PhysRevMaterials.2.013803
    =[1c1] 

[27] T. L. Jacobsen, M. S. Jørgensen, and B. Hammer, Phys. Rev. Lett. 120, 026102 (2018).
https://doi.org/10.1103/PhysRevLett.120.026102

[28] P. E. Dolgirev, I. A. Kruglov, and A. R. Oganov, AIPAdv. 6,085318 (2016).
https://doi.org/10.1063/1.4961886


[1c4] [25]のレビューを読む

"Perspective: Machine learning potentials for atomistic simulations"
J. Behler, J. Chem. Phys. 145, 170901 (2016).
J. Chem. Phys. 145, 170901 (2016)
https://doi.org/10.1063/1.4966192

II. STRUCTURAL DESCRIPTION にて、結晶構造の記述の方法を議論している。
A. The role of the descriptor
B. Descriptors for machine learning potentials
種類が多くて把握しきれない
特徴量の作成は、機械学習の手法そのものよりも重要な場合もあるほど
力を入れるのは当然か
？今から私が簡単に組めるのはない？


[1c4] 続編
AutoEncoder by Forest
Ji Feng, Zhi-Hua Zhou
https://arxiv.org/abs/1709.09018

オートエンコーダについて検索・学習
https://deepage.net/deep_learning/2016/10/09/deeplearning_autoencoder.html
オートエンコーダの核は次元削減である。
オートエンコーダはニューラルネットワークの一種で、情報量を小さくした特徴表現を獲得するためにある。

ニューラルネットワークのパラメータの初期値をランダムではなく、
オートエンコーダで訓練したものを用いるというアイデアが試された。

Greedy Layer-wise Trainingの場合は、まず教師なしデータで層のパラメータを1層1層順番に調整していく。
...
このようにオートエンコーダを用いてパーセプトロンの重みの初期値を予め推定しておくことを
事前学習(pre-training)という。
こうすると、完全なランダム値と比べて、勾配消失問題が起こる可能性が小さくなり、層を深くしても学習がうまく進むことが分かった。

では、オートエンコーダが今のディープラーニングを支えているのかというと、そうでもなさそうだ。
深層学習ライブラリKerasのオートエンコーダのチュートリアルには、
もう今では実用的な用途としてはめったに使われてないと書かれている。
オートエンコーダは画像のノイズ除去や可視化程度でしか利用目的がないとされている。

実は、その後ディープラーニングのアルゴリズムは改良を重ね、
事前学習をせずにランダム値を採用しても十分な精度が出るようになったのだ。

Wikipedia オートエンコーダ
オートエンコーダ（自己符号化器、英: autoencoder）とは、
機械学習において、ニューラルネットワークを使用した次元圧縮のためのアルゴリズム。
2006年にジェフリー・ヒントンらが提案した[1]。
線形の次元圧縮としては主成分分析があるが、オートエンコーダはニューラルネットワークを使用する。 

オートエンコーダは3層ニューラルネットにおいて、
入力層と出力層に同じデータを用いて教師あり学習させたものである。
バックプロパゲーションの特殊な場合と言える。


[1d] 考察・疑問
？構造最適化の機械学習をやるにあたり、他人の真似をする？自分で雑なものを考える？
？結晶構造のfinger-print, descripterは必須なのか？
？ニューラルネットワーク・ディープフォレストのような、ネットワークによる機械学習は、
　他の機械学習手法で同様の事をやっても性能は上がるのか？
？ハイパーパラメータを最適化せず、各ハイパーパラメータを持ったモデルを複数用意して
　アンサンブル学習は性能が上がるのか？
？ニューラルネットワークで、同じ層の中の人工ニューロンはどうやって、差別化している？
　ランダムにハイパーパラメータ割り振ってから調整しているとか？
　上述の通り、最初はそうだったらしいが、以降は事前学習を用いている



[1e] データベース更新
"High-pressure structures of yttrium hydrides"
Lu-Lu Liu et al 
2017 J. Phys.: Condens. Matter 29 325401
https://doi.org/10.1088/1361-648X/aa787d
Table 3. をそのままコピペで終わり

"Potential high-Tc superconducting lanthanum and yttrium hydrides at high pressure"
Hanyu Liu, Ivan I. Naumov, Roald Hoffmann, N. W. Ashcroft, and Russell J. Hemley
Proc Natl Acad Sci U S A. 2017 Jul 3; 114(27): 6990–6995.
https://doi.org/10.1073/pnas.1704505114
Table S1. をそのままコピペ+対称性調べて終わり
McMillanとEliashbergの２通り載っているが、McMillanのみ採用
