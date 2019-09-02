# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 15:41:37 2018

@author: Akitaka
"""

[1b2]
tipsに追加
クロスバリデーション/交差検証/CVのfold数は？
0ではないけど分散が小さいから、という理由だけで変数を削除してはいけない
Leave-one-outクロスバリデーションの２つのデメリット、からの解決方法
適当に描いてはダメ！実測値と推定値との正式なプロットを描くときの３つの注意点


[1b3]
例の総当たりプログラムに必要なスクリプトと関数を参考にする
0620/supportingfunctions.py

http://univprof.com/archives/16-06-09-3792550.html
例のtipsに追加した、以上のページとは若干ことなる部分がある
yyplotの範囲
上限: 最大値＋0.1(最大値ー最小値)　　ページ
上限: 最大値＋0.05(最大値ー最小値)　　ソースコード my_library.py


[1b4]
グラム行列からガンマの最適化 + OCSVMやSVMに実装(どちらか１つでいい)
グラム行列の計算は、説明変数が100でも可能

ref:
変数のネーミングが違うだけで、全部おなじ
0620/supportingfunctions.py
https://github.com/hkaneko1985/fastoptsvrhyperparams/blob/master/fastoptsvrhyperparams.ipynb
https://datachemeng.com/wp-content/uploads/variable_transform_ad.py


1つ目を基に、変数名を変えて、 optimize_gamma(X, gammas) を作成 @ my_libraryに追加
SVM+OCSVM+分類にて適用。
(1) SVMで最適化したγでOCSVM
(2) グラム行列の分散を最大化するγでOCSVM
結果は確かに異なる。どちらの方が正確というのは別にないというか、場合による？

Tc計算にも応用。
しかし、たまたま？γが一致してしまった！　→　以下、無理矢理γを別の数字に変えてテスト
複数のADの判定基準のどちらかを満たしたもののみをcsvに書き込む
df_in_ = df[(df.AD1 ==  1) | (df.AD2 ==  1)]
これを使えば、kNNの判定基準も比較できる！

example:
my_library.py
test0_SVM_OCSVM_DCV_clf.py
test2_Tc_SVM_OCSVM_DCV.py

ref:
pandasで複数条件のand, or, notから行を抽出（選択）
https://note.nkmk.me/python-pandas-multiple-conditions/




[1c] 06/15 (Anaconda使い始め)からの総括　続き
課題だけでなく、成果もまとめる
今まで、test0,tes1,のみだったファイル名は、なるべく内容も合わせて記載する
test1_16rgr.pyなど

0615~06/29,
0702~08/01まで完了


0705
[1e1] 物理量を取得
欠損値のチェック
  test2.py
mendeleevの値と、wikipediaの値と、明らかに合っていない！
単位がどちらもkJ/molのはずなのに全然合わない
[todo] test3のデータベースも要チェック
    0705/test3_atomic_data.py



0710job
[1g] jobスクリプトのテスト
2018/0312/test6/job
2018/0424/test2.job
2018/0425/test1.job
2018/0426/test1.job
2018/0426/test2.job
2018/0427/test2.job
あたりを参考に
example:
618551/test0.job



0713test6_7clf_dcv.py
[1g] ダブルクロスバリデーション/二重交差検証/DCV + 21種類のクラス分類方法テスト(総当たり)
ref:
0706/test4.py
0707/test1_dcv.py
0707DCVの目的(回帰係数？推定性能？モデル選択？)やDCV予測値について詳しい
0711/test1_dcv.py
→DCVと合流
0713/test6_7clf_dcv.py




0711test2_21clf.py
[1g2] 21種類のクラス分類方法テスト(総当たり)(Eg >0 か Eg = 0の分類)
ref:
0625/test5.py
0626/test7.py
0702/test1.py
0703/test3.py
0711/test2_21clf.py




711test5_7clf_dcv.py
モデルごとに違うハイパーパラメータで最適化
[1g5] DCV＋複数モデルでモデルごとの評価
    21個の分類手法を試していたが、多すぎるので、論文を参考に抜粋する
    モデルごとにハイパーパラメータが違う
    ３重入れ子の辞書を作って、まとめることに成功
ref:
0711/test5_7clf_dcv.py



0713test5_Eg_RFC_RFR.py
分類+回帰
[1f2] EgでRFC+RFRのテスト
ref:
0705,0706,
0711/test6_Eg_RFC_RFR.py
0712/test1_Eg_RFC_RFR.py
0713/test5_Eg_RFC_RFR.py




test0_SVM_OCSVM_DCV_clf.py
test0_SVM_OCSVM_DCV_rgr.py
SVM+OCSVMの開発
ref:
0713, 0716, 0717, 0718, 0724, ...
→DCVと合流
0726test0_SVM_OCSVM_contour.py
等高線を使った図示はここで終了



test0_kNN_AD_DCV_clf.py
test0_kNN_AD_DCV_clf.py
ref:
0726, 0727



0521test2_convert_cf2parameter.py
0521/test2.py tc.csvファイルで化学式を原子番号と個数に変換
0724/test2_Tc_SVM.pyで逆変換
0521/test3.py SVMでtc予測



test1_to_csv.py
ref:
0724 ほぼ完成



test2_Tc_kNN_AD_DCV.py
ref:
0727, ...



test2_Tc_SVM_OCSVM_DCV.py
ref:
0426/test1.py が重要
    cvの数、cv=KFold, Shufflesplitなどの違いに言及
    Shufflesplitがベストと結論
    また、ShuffleSplitとscalingを併用することで性能が向上したとも書いてある
0427/test1.py
    スケーリングをMinMax, Standardのどちらにするかの比較
0511/test1.py = 0427/test1.py
    発表前に軽く計算しただけ
07/18 水素化物Tc計算
07/24以降 アプデ