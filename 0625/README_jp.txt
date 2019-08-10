[1c] t-SNEの簡単なサンプルを実行
C:\Users\Akitaka\Downloads\python\20180623/test1.ipynb PCA
C:\Users\Akitaka\Downloads\python\20180623/test2.ipynb t-SNE

scikit-learnでt-SNE散布図を描いてみる
http://inaz2.hatenablog.com/entry/2017/01/24/211331
PCAと比較したもの。差は歴然

Python: 多様体学習 (Manifold Learning) を用いた次元縮約
https://blog.amedama.jp/entry/2017/12/09/142655
C:\Users\Akitaka\Downloads\python\20180623/test3.ipynb digitのテスト
C:\Users\Akitaka\Downloads\python\20180623/test4.ipynb 比較
PCA, MDS, Isomap, LocallyLinearEmbedding, Laplacian Eigenmaps と比較したもの。差は歴然


[1d] ギャップを機械学習してみる　分類編
Egの数値を「回帰」ではなく、金属か絶縁体かを「分類」する
ref:06/22

C:\Users\Akitaka\Downloads\python\20180625\test5.py
方法
ハイパーパラメータはデフォルト
X:ABO2の、A,Bの原子番号NA, NBの２つのみ
y:金属(1)か絶縁体(2)か
データは全部 X_train とする
手法は以下の２１全部実行
[実行用プログラム公開] ２１の判別分析(二クラス分類)手法を一気に実行して結果を比較する！
ダブルクロスバリデーションによる評価付き (Python言語)
https://note.mu/univprof/n/n38855bb9bfa8

結果
                                    metrics.accuracy_score(y, y_pred))
Linear Discriminant Analysis        0.719
Linear Support Vector Machine       0.750
Non-Linear Support Vector Machine   0.854
Quadratic Discriminant Analysis     0.760
k-Nearest Neighbor Classification   0.781
Gaussian Naive Bayes                0.740
Decision Tree                       1.000
Random Forests                      0.969
Gaussian Process Classification     0.729
Bagging[LDA]                        0.708
Bagging[LSVM]                       0.729
Bagging[NLSVM]                      0.885
Bagging[QDA]                        0.750
Bagging[kNNC]                       0.781
Bagging[NB]                         0.719
Bagging[DT]                         0.958
Bagging[GPC]                        0.990
AdaBoost[LSVM]                      0.729
AdaBoost[NLSVM]                     0.729
AdaBoost[NB]                        0.740
AdaBoost[DT]                        1.000

※18. LSVMに基づくAdaptive Boosting (AdaBoost[LSVM])のみ、エラーが出た
ValueError: BaseClassifier in AdaBoostClassifier ensemble
 is worse than random, ensemble can not be fit.
上の18.の値はSVLのまま

全部絶縁体に分類したものもある。それでも元々絶縁体が多いと精度が上がってしまう！
→AUC、ROC曲線を使うべきでは ref:04/10-11