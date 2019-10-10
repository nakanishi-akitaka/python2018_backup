# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 10:09:39 2018

@author: Akitaka
"""

[1b] 水素化物Tc計算
[1b1] 過去まとめなど
注意
20181113\Tc_new.py -> 20181114\Tc_6model.py

ファイル統一過去のまとめ
20181026:EN, RR, LASSO, kNN
20181105:EN, RR, LASSO, kNN, RF, SVR
20181107:EN, RR, LASSO, kNN, RF, SVR + GPR(別ファイル)
20181113:仕様を大幅に変更


[1b2] PLSの追加 → なしとする(説明変数を増やした後でやるのが良いかもしれない)
理由1 サンプルプログラムが１つしかないので、あまり重要そうでない
https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html

理由2 scikit-learnのチートシートに載っていないので、～
http://neuro-educator.com/mlcontentstalbe/


[1b3] GBoostを追加(1017で個別のは作成していた)

ref:06/29
https://mail.google.com/mail/u/0/#sent/KtbxLzGcCFBMNnCsPWdXPBjgRPkCZFtXmL
XGBoost、アンサンブル学習、ブースティングなどについて詳しく調べた
XGBoost = Gradient Boosting（勾配ブースティング）の高速(約10倍)なC++実装

ref:10/17 実装した日
https://mail.google.com/mail/u/0/#sent/KtbxLzGDdjkMfdxXxfMhNRJhstrTtWVRSq

GBoostであればscikit-learnで用意されている
XGBoostをインストールする場合は、以下を参考に
    ref:
    http://yag.xyz/blog/2015/08/08/xgboost-python/
    http://wolfin.hatenablog.com/entry/2018/02/08/092124
    http://tekenuko.hatenablog.com/entry/2016/09/22/220814
    https://qiita.com/TomokIshii/items/290adc16e2ca5032ca07

計算結果
runfile('C:/Users/Akitaka/Downloads/python/1114/Tc_7model.py',
wdir='C:/Users/Akitaka/Downloads/python/1114')
Gradient Boosting
Best parameters set found on development set:
{'n_estimators': 500}
C:  RMSE, MAE, R^2 =  9.015,  6.433, 0.976
CV: RMSE, MAE, R^2 = 28.591, 15.557, 0.754
P:  RMSE, MAE, R^2 = 79.652, 61.733, 0.000
DCV:RMSE, MAE, R^2 = 29.360, 17.965, 0.739 (ave)
DCV:RMSE, MAE, R^2 =  2.179,  1.668, 0.040 (std)
rnd:RMSE, MAE, R^2 = 55.424, 40.698, 0.074 (ave)
rnd:RMSE, MAE, R^2 =  0.495,  0.387, 0.017 (std)
88.49 seconds

formula,P,Tc,AD
ZrH10,0,299,1
SrH10,0,299,1
YH10,0,299,1
RbH10,0,299,1
YH10,0,295,1
RbH10,0,295,1
ZrH10,0,295,1
SrH10,0,295,1
YH10,0,280,1
SrH10,0,280,1

[todo]->[stop] GBoostingについて調べて、以下の疑問に答える
    ref:20181017
    似たようなTcなのは、RFと同様
    しかし、今までの(kNN, SVR, RF)と異なり、XH3ではない。
    何が原因？
アップデートしたデータベースを使った、今回の学習ではだいたい、XH9~H10が多い
データベースにもよる問題らしいので、これ以上の考察は無し



[1b4] 過去のパラメータ作成について
summary/0705test1_Tc_SVM_make_parameters.py
原子のデータから、パラメータを計算
ref:
https://arxiv.org/abs/1803.10260

summary/0705test3_atomic_data.py
原子データcsvの作成


[1b5] 説明変数を変えて計算
取りあえず、簡単に計算できるものからテスト
方法はkNN。testする範囲を絞るなどして、軽量化。
変数：原子番号、原子番号のルート、個数
パラメータの数が増えたときに対応できないバグがあったので修正。

"A Data-Driven Statistical Model for Predicting the Critical Temperature
 of a Superconductor"
https://arxiv.org/abs/1803.10260
を参考もとい、そのままマネした
各種物理量を基に計算したパラメータを用いる
詳細は0705のメールや、論文参照

！パラメータが増えた分、時間がかかる
1.質量からパラメータを作成した場合の結果
Best parameters set found on development set:
{'n_neighbors': 1}
C:  RMSE, MAE, R^2 =  8.819,  4.420, 0.977
CV: RMSE, MAE, R^2 = 30.869, 17.383, 0.713
P:  RMSE, MAE, R^2 = 31.985, 23.315, 0.000
DCV:RMSE, MAE, R^2 = 32.816, 19.355, 0.672 (ave)
DCV:RMSE, MAE, R^2 =  3.629,  1.797, 0.070 (std)
rnd:RMSE, MAE, R^2 = 51.787, 38.248, 0.192 (ave)
rnd:RMSE, MAE, R^2 =  0.962,  0.773, 0.030 (std)

昨日の、原子番号と個数から～
{'n_neighbors': 1}
C:  RMSE, MAE, R^2 =  9.043,  4.504, 0.975
CV: RMSE, MAE, R^2 = 22.845, 14.049, 0.843
P:  RMSE, MAE, R^2 = 58.395, 35.507, 0.000
DCV:RMSE, MAE, R^2 = 30.493 (+/-2.053), 17.645 (+/-1.220), 0.719 (+/-0.037)
y-r:RMSE, MAE, R^2 = 54.101 (+/-0.710), 39.979 (+/-0.729), 0.118 (+/-0.023)
291.46 seconds

DCVの性能はむしろ下がってしまった
昨日から、n_neighbors=1が最適な値なのが気になる
一番近いデータのをそのままマネしてるだけになっている！


2.1+第一イオン化エネルギー
{'n_neighbors': 1}
C:  RMSE, MAE, R^2 =  8.970,  4.480, 0.976
CV: RMSE, MAE, R^2 = 28.304, 15.058, 0.759
P:  RMSE, MAE, R^2 = 25.705, 18.041, 0.000
DCV:RMSE, MAE, R^2 = 27.180, 15.750, 0.777 (ave)
DCV:RMSE, MAE, R^2 =  1.528,  1.124, 0.025 (std)
rnd:RMSE, MAE, R^2 = 51.454, 38.111, 0.202 (ave)
rnd:RMSE, MAE, R^2 =  0.816,  0.940, 0.025 (std)
1301.77 seconds


3.2+原子半径
{'n_neighbors': 1}
C:  RMSE, MAE, R^2 =  8.869,  4.422, 0.976
CV: RMSE, MAE, R^2 = 23.658, 14.021, 0.831
P:  RMSE, MAE, R^2 = 35.214, 20.416, 0.000
DCV:RMSE, MAE, R^2 = 24.880, 14.406, 0.809 (ave)
DCV:RMSE, MAE, R^2 =  3.669,  1.554, 0.058 (std)
rnd:RMSE, MAE, R^2 = 51.621, 38.212, 0.197 (ave)
rnd:RMSE, MAE, R^2 =  0.816,  0.758, 0.025 (std)
509.15 seconds


4.3+融点
{'n_neighbors': 1}
C:  RMSE, MAE, R^2 =  8.955,  4.469, 0.976
CV: RMSE, MAE, R^2 = 25.927, 14.887, 0.797
P:  RMSE, MAE, R^2 = 23.783, 17.328, 0.000
DCV:RMSE, MAE, R^2 = 26.819, 15.736, 0.781 (ave)
DCV:RMSE, MAE, R^2 =  2.606,  1.268, 0.041 (std)
rnd:RMSE, MAE, R^2 = 50.833, 37.433, 0.221 (ave)
rnd:RMSE, MAE, R^2 =  0.619,  0.627, 0.019 (std)
183.13 seconds

5.4+価電子数
{'n_neighbors': 1}
C:  RMSE, MAE, R^2 =  8.782,  4.399, 0.977
CV: RMSE, MAE, R^2 = 28.824, 17.433, 0.750
P:  RMSE, MAE, R^2 = 29.118, 20.844, 0.000
DCV:RMSE, MAE, R^2 = 27.404, 16.107, 0.770 (ave)
DCV:RMSE, MAE, R^2 =  3.413,  1.674, 0.056 (std)
rnd:RMSE, MAE, R^2 = 51.551, 38.042, 0.199 (ave)
rnd:RMSE, MAE, R^2 =  0.741,  0.669, 0.023 (std)
219.27 seconds

6.5+密度
{'n_neighbors': 5}
C:  RMSE, MAE, R^2 = 21.441, 13.362, 0.862
CV: RMSE, MAE, R^2 = 24.469, 14.142, 0.820
P:  RMSE, MAE, R^2 = 26.770, 22.003, 0.000
DCV:RMSE, MAE, R^2 = 26.714, 15.825, 0.784 (ave)
DCV:RMSE, MAE, R^2 =  1.602,  0.871, 0.026 (std)
rnd:RMSE, MAE, R^2 = 51.060, 37.764, 0.214 (ave)
rnd:RMSE, MAE, R^2 =  1.275,  1.340, 0.039 (std)
214.50 seconds

7.6+熱電導
{'n_neighbors': 1}
C:  RMSE, MAE, R^2 =  8.097,  4.126, 0.980
CV: RMSE, MAE, R^2 = 28.760, 17.204, 0.751
P:  RMSE, MAE, R^2 = 34.478, 23.327, 0.000
DCV:RMSE, MAE, R^2 = 24.428, 14.369, 0.818 (ave)
DCV:RMSE, MAE, R^2 =  2.832,  1.289, 0.043 (std)
rnd:RMSE, MAE, R^2 = 51.574, 38.384, 0.199 (ave)
rnd:RMSE, MAE, R^2 =  0.661,  0.912, 0.020 (std)
184.05 seconds

8.7+電気陰性度
Best parameters set found on development set:
{'n_neighbors': 1}
C:  RMSE, MAE, R^2 =  8.112,  4.150, 0.980
CV: RMSE, MAE, R^2 = 26.995, 15.217, 0.780
P:  RMSE, MAE, R^2 = 31.423, 21.277, 0.000
DCV:RMSE, MAE, R^2 = 25.775, 15.219, 0.799 (ave)
DCV:RMSE, MAE, R^2 =  1.348,  1.034, 0.021 (std)
rnd:RMSE, MAE, R^2 = 51.133, 37.854, 0.212 (ave)
rnd:RMSE, MAE, R^2 =  0.623,  0.521, 0.019 (std)
270.72 seconds

注意：電気陰性度、密度、熱伝導率は、一部の原子のデータが無かったため、適当な値に置き換えた



[1c] todo処理
[todo] -> [stop]
    回帰・分類の総当たり＋ハイパーパラメータの最適化する？
    どのハイパーパラメータを最適化するべきかは金子研究室のHPを参考にする？
色々テストして、予測性能の高いものだけをピックアップする
ただし、総当たりのために最小二乗法などの明らかに精度の低いものをやる必要性はない

[todo] -> [done]
    DCVやy-randamizationでのスケーリングは？
    param_gridを２通り(pipe使用の有無)用意するのを避けるため、
    スケーリングを使用しない(前処理なし学習のみ)場合でもpipeを使うことにする
pipeなし。予めスケーリングすることにした。

[todo] -> [doing]
    水素化物Tc予測プログラム 更新の余地
    * 説明変数の増加 + PCAなど
-> 実行中

    * 何度も行って、目的変数の予測値の平均と分散を計算する
-> 中止
ランダムフォレストのようなものでないと大して意味がない
誤差を見積もりたいなら、DCVで十分。やっていることが近いため。

    * Tc=0Kとの分類＋回帰では？
-> 予定
そもそもTc = 0Kのデータを集める
安定に存在できない組成比があるのをTc=0Kとしてデータベースにいれる？

    * インプットアウトプットを下のプログラムに倣って、train, predictionにする？
    https://note.mu/univprof/n/n7d9eb3ce2c74
    https://note.mu/univprof/n/n38855bb9bfa8
    プログラム・コードを実行するためのcsvファイルのデータ形式・サンプルデータ(MATLAB, R, Python)
    https://note.mu/univprof/n/n694487325cb1
-> OK


[todo] -> ????
考察
kNNやOLSでもいいから、実現できないか？
個性的な人工知能をつくる
https://datachemeng.com/uniqueartificialintelligence/
1.アウトプットの信頼性も一緒に返してくれる
    ADorデータ密度がその一種。GPRやアンサンブル学習なら、予測値と同時に分散を出力できる。
2.どんなデータがあれば、信頼性が上がるか教えてくれる
    ADでないが、ADに近い領域を表示できればいいのでは？
3.逆解析で複数のアウトプットを返すときに、補足情報によって順位づけしてくれる
    逆解析する場合、異なるxに対して同じyが出力されることがある
    その際、1.によりyの信頼度も出力すれば、それでランク付けできる
4.どうしてそのアウトプットになったか教えてくれる
    補完みたいな手法、ノンパラメトリック回帰の場合は、近くのデータがそうだから、としか言えない(kNNなど)
        強いて言えば、近く(と機械が判断した)データを表示するぐらいか
    モデルを設定する、パラメトリック回帰の場合は、理解しやすい
    ただし、ニューラルネットワークのような複雑すぎるものは厳しい

1.4.ができるかは手法次第な部分がある
1.ができれば、3.はほぼ自動的にできる
AD, データ密度が計算できれば、2.はすぐできる
一応、kNNでそれなりのことはできるっぽい



[1d] 考察
機械学習って、結局はただの補完じゃないの？
線形であれ非線形であれ
線形補完の範囲をデータのある領域全部でやる　→　最小二乗法ともいえる

20180919のPLS, LWPLS考察より
PLSは単純な線形モデル
LWPLSは、予測したいデータの近くの学習データをなるべく再現するよう線形モデルを組む
小さい区間での線形補完を繰り返しやってるようなものでは？

少なくとも、最初からモデルの形を仮定しないノンパラメトリック回帰はほぼ補完と同じ印象
モデルの形を仮定するの、パラメトリック回帰は、全区間での補完とも言える？
SVR：カーネル関数の形で補間　ガウス関数など
ランダムフォレスト：ある範囲にあるデータを線形y=constantで近似したようなもの

結局は補完なのであれば、特別目新しい発見はないのか？
画像識別のような、説明変数が多いもので、
かつ、単純な補完が難しいものであれば、機械学習である意味がありそう



