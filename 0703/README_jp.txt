【機械学習】モデル評価・指標についてのまとめと実行( w/Titanicデータセット)
https://qiita.com/kenmatsu4/items/0a862a42ceb178ba7155
  test1.py

ROC曲線を描く方法
http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
を用いて計算
  test2.py

ギャップを機械学習してみる　分類編
Egの数値を「回帰」ではなく、金属か絶縁体かを「分類」する
ref:06/22, 25, 26, 07/02
  test3.py
https://qiita.com/kibinag0/items/1a29db61fcb8c527d952
ここを参考に、Accuracy, Precision, Recall, F1-score, AUCでCVを行う

ROC曲線のサンプル
http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
  test4.py
マルチクラス分類
1.１つのクラスについてのROC曲線
2.全てのクラスについてのROC曲線


http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html
  test5.py
交差検証/CVした時の、各FoldについてのROC曲線


Pythonで実践編
【Pythonで決定木 & Random Forest】タイタニックの生存者データを分析してみた
http://www.randpy.tokyo/entry/python_random_forest
  test6.py

論文追試
https://arxiv.org/pdf/1803.10260.pdf
Table 1,2から、80個の説明変数を作成するテスト
  test7.py