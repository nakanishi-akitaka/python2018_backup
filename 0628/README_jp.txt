scikit-learn で機械学習
https://pythondatascience.plavox.info/scikit-learn

scikit-learn でトレーニングデータとテストデータを作成する
  test1.py
scikit-learn で線形回帰 (単回帰分析・重回帰分析)
  test2.py
scikit-learn でクラスタ分析 (K-means 法)
  test3.py
scikit-learn で決定木分析 (CART 法)
  test4.py
scikit-learn でクラス分類結果を評価する
  test5.py
scikit-learn で回帰モデルの結果を評価する
  test6.py
scikit-learn に付属しているデータセット
  test7.py

決定木グラフを作図するため、Graphvizをインストール
https://graphviz.gitlab.io/download/

pydotplusもpipでインストール
https://pypi.python.org/pypi/pydotplus



[1c] 分類の評価指標について
混同行列　wikipedia
https://en.wikipedia.org/wiki/Confusion_matrix
色んなパターンがある！

機械学習で使う指標総まとめ(教師あり学習編)
http://www.procrasist.com/entry/ml-metrics
どの評価を使うかは、目的によると言える
例1:果物をみかんorNotで分類する場合は、
　正しく分類できているかどうかが大事なので、正解率/Accuracy
例2:がん検診は、ガンを健康と誤診するのを一番避けたい
　逆に健康をガンと診断しても構わない（ガンと診断した場合はどうでもいい）
　ので、再現率/Recallを１にすることが大事
まとめ
* 特定の閾値を設定せずに評価したい場合 -> ROC, AUC
* 特定の閾値を設定して評価したい場合
** Aである、Aでないの重要度が一緒の場合 Accuracy
** Aであると言う事実が重要な場合
*** Aであると予測して、実際にAであった人の割合が重要
    (試験、再検査にコストがかかるなど)な場合 -> Precision
*** 実際にAである人をきちんとAと予測できているか(Recall)が
    重要(検知漏れが許されない) -> Recall
*** 両方を見たい場合 -> (weighted) F-measure



[1d] 論文ではどんな評価指標を使っているか？調べる
"Prediction of Low-Thermal-Conductivity Compounds with
 First-Principles Anharmonic Lattice-Dynamics Calculations and Bayesian Optimization"
Atsuto Seko, Atsushi Togo, Hiroyuki Hayashi, Koji Tsuda, Laurent Chaput, and Isao Tanaka
Phys. Rev. Lett. 115, 205901 - Published 10 November 2015
DOI:https://doi.org/10.1103/PhysRevLett.115.205901
不明

"Multiclassification Prediction of Enzymatic Reactions for Oxidoreductases and Hydrolases
 Using Reaction Fingerprints and Machine Learning Methods"
Yingchun Cai , Hongbin Yang , Weihua Li , Guixia Liu, Philip W. Lee, and Yun Tang* 
J. Chem. Inf. Model., 2018, 58 (6), pp 1169-1181
DOI: 10.1021/acs.jcim.7b00656

モデルの評価 = Precision, Recall, F1-score
p1170
> In addition, three metrics P (Precision), R (Recall), and F (F1-score)
>  were used to evaluate the performance of each model


"Machine learning modeling of superconducting critical temperature"
https://arxiv.org/abs/1709.02727
超伝導での機械学習も、すでに出た。
？　私のやることは何がある？
12000+SuperConデータベース
1.組成のみを特徴量として、Tc 10K以上と未満と2通りに分けるモデルを構築　正解率92%
2.それをTcを具体的に予測するモデルに改良
ほかの特徴量をAFLOWから取得することでさらに改良
約30個の候補(非銅酸化物、非鉄酸化物)を得た＝Table 3
　Tc > 20Kだが具体的には不明
ランダムフォレスト回帰
ref:2017/09/11

モデルの評価 = Accuracy, Precision, Recall, F1-score
p4
> Hypothetically, if 95% of the observations in the
> dataset are in the below-Tsep group, simply classifying
> all materials as such would yield a high accuracy (95%),
> while being trivial in any other sense. To avoid this potential
> pitfall, three other standard metrics for classification
> are considered: precision, recall, and F1 score. They
> are defined using the values tp, tn, f p, and fn for the
> count of true/false positive/negative predictions of the model:


上の論文を引用した論文
https://www.researchgate.net/publication/319622538_Machine_learning_modeling_of_superconducting_critical_temperature
"A Data-Driven Statistical Model for Predicting the Critical Temperature of a Superconductor"
https://arxiv.org/abs/1803.10260

使用した物理特性８つ
 = Atomic Mass, First Ionization Energy, Atomic Radius, Density,
   Electron Affinity, Fusion Heat, Thermal Conductivity, Valence  
Table 1: This table shows the properties of an element which are used for creating features to
predict Tc.

↑から作成した特徴量
例：Thermal Conductivity の平均、重み付き平均、相乗平均など
Table 2: This table summarizes the procedure for feature extraction from material’s chemical formula

評価指標 = RMSE, R^2
> Our XGBoost model gives good predictions: an out-of-sample error of about 9.5 K
> based on root-mean-squared-error (rmse), and an out-of-sample R2 values of about 0.92. T


マテリアル・インフォマティクス論文読み＋議論
"Accelerated Materials Design of Lithium Superionic Conductors Based
on First-Principles Calculations and Machine Learning Algorithms"
Koji Fujimura
Adv Energy Mater 3. 980 (2013)
DOI: 10.1002/aenm.201300060
ref:2016/12/06

評価指標 ???
> The variance of the Gaussian kernel, the regularization
> constant and forms of independent variables were optimized
> by minimizing the prediction error estimated by the bootstrap-
> ping method. [36] The prediction error of the optimized SVR for
> log σ is 0.373. 


準安定構造についての機械学習
"The thermodynamic scale of inorganic crystalline metastability"
Wenhao Sun et al., Science Advances  18 Nov 2016:Vol. 2, no. 11, e1600225
DOI: 10.1126/sciadv.1600225


