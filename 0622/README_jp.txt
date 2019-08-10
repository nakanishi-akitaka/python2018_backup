[1b] データ変換
C:\Users\Akitaka\Downloads\python\20180622\test1.py
tmxo2_gap.csvを読み込む
→CuHO2を[47, 1, 8, 1, 1, 2]のように変換
→test1.csvに書き込む

pandasを使うと、どちらも1行で終わる
データ形式変換などは必要だが

参考
https://pythondatascience.plavox.info/pandas/データフレームを出力する
https://pythondatascience.plavox.info/pandas/csvファイルの読み込み



[1c] ギャップを機械学習してみる　回帰編
C:\Users\Akitaka\Downloads\python\20180622\test2.py
ref:06/20, 20180620/test1.pyを参考に、回帰手法総当たり
→RNRのみ、なにかエラー？が起きる 
　y_predでNaNがでる。
→NaNを-1に置換する
ref
NumPyの配列ndarrayの欠損値np.nanを他の値に置換
https://note.nkmk.me/python-numpy-nan-replace/

方法
ハイパーパラメータはデフォルト
X:ABO2の、A,Bの原子番号NA, NBの２つのみ
y:エネルギーギャップ
データは全部 X_train とする

結果
スコアはRMSE, MAE, RMSE/MAE, R2の順番
LR  0.752, 0.590, 1.275, 0.023
DTR 0.000, 0.000, 0.000, 1.000
RFR 0.220, 0.169, 1.302, 0.917
OMP 0.753, 0.593, 1.269, 0.020
RAN 0.870, 0.633, 1.373, -0.308
BR  0.754, 0.598, 1.262, 0.017
BGM 1.098, 0.792, 1.387, -1.083
KNR 0.679, 0.515, 1.317, 0.204
RNR 0.524, 0.358, 1.464, 0.526
PLS 0.752, 0.589, 1.275, 0.023
SVL 0.765, 0.577, 1.326, -0.012
SVR 0.433, 0.264, 1.642, 0.676
LAS 0.754, 0.598, 1.261, 0.017
EN  0.752, 0.594, 1.267, 0.022
RR  0.752, 0.590, 1.275, 0.023
GPR 0.000, 0.000, 1.328, 1.000
TSR 0.767, 0.606, 1.266, -0.018
有力なのは、DTR, RFR, RNR, SVR, GPR



[1d] 分類計算のテスト
C:\Users\Akitaka\Downloads\python\20180622\test3.py

このサンプルをコピペしてみる
http://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_probas.html
ロジスティック回帰、単純ベイズ分類器、ランダムフォレスト、の３つ
また、それらの多数決を実行する関数もある


[1b2] データ変換その２
C:\Users\Akitaka\Downloads\python\20180622\test1.py

データを読み取るときに、
Eg>0 -> 2
Eg=0 -> 1
と変換して、test1_cls.csvに書き込む
 

[1e] ギャップを機械学習してみる　分類編
Egの数値を「回帰」ではなく、金属か絶縁体かを「分類」する

方法
決定木のみ
ハイパーパラメータはデフォルト
X:ABO2の、A,Bの原子番号NA, NBの２つのみ
y:絶縁体=1, 金属=2
データは全部 X_train とする。検証・予測なし

評価方法は
python_work_fs01/2018/0406/test2.pyを参考に色々書く

結果
metrics.confusion_matrix
[[26  0]
 [ 0 70]]
完全に分類できた
過学習の可能性もあるので、一概に喜べないが


[1f] 論文読み
ref:06/05 <div>
金子教授のtwitterより
https://twitter.com/hirokaneko226/status/1002123554463272960
加水分解酵素と酸化還元酵素に関する代謝反応をクラス分類する論文。

Multiclassification Prediction of Enzymatic Reactions
 for Oxidoreductases and Hydrolases
 Using Reaction Fingerprints and Machine Learning Methods
Yingchun Cai et al. (China group)
J. Chem. Inf. Model.
DOI: 10.1021/acs.jcim.7b00656
※巻、ページなどは未定。電子版のみ公開中？

図１によると、
 反応→反応物＋生成物からなる→それぞれについてAP,Mogan2,TT,PFがある
 →その4つからRDF,SRF,TRFを計算？→この3つを反応フィンガープリントとする
 KEGG→反応フィンガープリントのデータ入手→trainingとtestに分ける
 Rhea(データベース)→反応～のデータ入手→validationにする
 trainingから機械学習でモデル作成
 チューニングしてbetterモデルに
 testでoptimalモデルに
 optimalモデルをvalidationで検証

クラス分類手法は７種類で検討。
＝decision tree (DT), k-nearest neighbors (k-NN), logistic regression (LR),
 naive Bayes (NB), neural network (NN), random forest (RF), and
support vector machine (SVM)

transformation reaction fingerprint でロジスティック回帰・ニューラルネットのケースが性能が良かった。
</div>
