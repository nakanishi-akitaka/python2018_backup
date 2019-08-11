[1b] サイトで勉強２　Python でデータサイエンス
https://pythondatascience.plavox.info/

Pandas でデータフレームを扱う
https://pythondatascience.plavox.info/pandas

Pandas でデータフレームを作ってみよう
  test1.py
Pandas のデータフレームを確認する
  test2.py
Pandas でデータフレームから特定の行・列を取得する
  test3.py
Pandas のデータフレームに行や列 (カラム) を追加する
  test4.py
Pandas のデータフレームの特定の行・列を削除する
  test5.py
Pandas のデータフレームの行・列の長さを確認する
  test6.py
Pandas のデータフレームの行⇔列を入れ替える
  test7.py
Pandas のデータフレームをソートする
  test8.py
Pandas でデータフレームの結合 (マージ, JOIN)
  test9.py
Pandas で CSV ファイルやテキストファイルを読み込む
  test10.py
Pandas のデータフレームを CSV ファイルやテキストファイルに出力する
  test11.py



[1b2] トラブルシューティング
毎回警告がでる
C:\Users\Akitaka\Anaconda3\lib\site-packages\spyder
\widgets\variableexplorer\utils.py:414: 
FutureWarning: 'summary' is deprecated and will be removed in a future version.
  display = value.summary()

該当ファイルを変更
old: display = value.summary()
new: display = value._summary()
ref:
https://github.com/spyder-ide/spyder/issues/7312



[1c] ROC曲線、AUC
【機械学習】モデル評価・指標についてのまとめと実行( w/Titanicデータセット)
https://qiita.com/kenmatsu4/items/0a862a42ceb178ba7155

test12.py
実行完了


[1c2]
ROC曲線とAUCについて定義と関係性をまとめたよ
https://qiita.com/koyamauchi/items/a2ed9f638b51f3b22cd6
roc_auc_score()のサンプル実行→エラー
tet13.py

http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
を参考に書き直し

[1c3]
評価指標いろいろ
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
test14.py


ref:04/10
https://mail.google.com/mail/u/0/#sent/LXphbRLrghxkrJntgFpbqNCtZspVcwXcxhpPbMnBxvV

ref:04/11
https://mail.google.com/mail/u/0/#sent/RdDgqcJHpWcvcDjPMnwQkJtsXTSDpJwzPDdxXTbtqTZV

読むべき？
http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html


[1c2] トラブルシューティング
※.ix → .ilocに変更
C:/Users/Akitaka/Downloads/python/0627/test12.py:92: DeprecationWarning: 
.ix is deprecated. Please use
.loc for label based indexing or
.iloc for positional indexing

See the documentation here:
http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated


フォーマット微修正
old: print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
new: print("Accuracy: {0:.2f} (+/- {1:.2f})".format(scores.mean(), scores.std() * 2))


警告文より
ライブラリの位置変更
* cross_validation
* grid_search

属性変更
grid_scores_
→
cv_results_
すると、おかしくなるのでやっぱりやめ
