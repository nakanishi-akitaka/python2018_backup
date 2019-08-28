# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 09:19:06 2018

@author: Akitaka
"""

[1a2] 
k-NNでの信頼度の計算は？(平均と標準偏差、多数決の割合など)
信頼度と適用範囲は別？
    適用範囲の中でも信頼度はバラバラ
    信頼度がある閾値以上なのがADでは？
        全く違う！
        信頼度＝1と0の割合(分類), 標準偏差(回帰)　→　どっちも標準偏差！
        適用範囲＝距離の平均
信頼度　分類なら.predict_proba(X)で、[0,1]それぞれの可能性(信頼度)が返ってくる
適用範囲　分類でも回帰でもNearestNeighbors.kneighbors(X)から自分で計算できそう


[1a3] プログラムを組む
SVM+OCSVM+DCVの、kNNバージョン
OCSVMはADだけだが、kNNはAD＋信頼度
example:
test4_kNN_AD_DCV_clf.py

信頼度
y_reli = np.absolute(gscv.predict_proba(X_test)[:,1]-0.5)+0.5

適用範囲
neigh = NearestNeighbors(n_neighbors=5)
neigh.fit(X_train)
dist = np.mean(neigh.kneighbors(X_test)[0], axis=1)
thr = dist.mean() - dist.std()
y_appd = 2 * (dist > thr) -1

※閾値は、3σ方法に基づき、μ-3σにしようとしたが、サンプル数10万(をtrain,testに分割)
でもADの外が0になるので、あまり意味がない？と思い　μ-σとした。

ref:
http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
https://note.nkmk.me/python-numpy-ndarray-sum-mean-axis/
https://algorithm.joho.info/programming/python/numpy-allclose/




[1c1] 水素化物のTc予測プログラムアップデート
example:
test1_to_csv.py
example:
test2_Tc_SVM_OCSVM_DCV.py

* OCSVMでAD調べる
結果にも出力 AD == 1 のみを抽出して出力
formula,P,Tc,AD
H3S,200,191,1
H3S,250,182,1
H3S,300,178,1
BiH6,300,91,1
BiH6,250,86,1
BiH6,200,80,1
H5S2,150,70,1
H2S,150,55,1
H5S2,100,50,1
BeH2,250,38,1
HS,300,24,1
HS2,200,24,1
HS2,250,15,1
HO2,100,11,1
HO2,250,7,1
TcH2,300,7,1
HO2,200,6,1
HO2,300,5,1
※テスト計算なので、探索範囲を小さくしている

* DCVを使って、予測性能を推定する
  ave, std of accuracy of inner CV: 0.325 (+/-0.059)
  ave, std of accuracy of inner CV: 0.386 (+/-0.029)
  ave, std of accuracy of inner CV: 0.350 (+/-0.056)
  ave, std of accuracy of inner CV: 0.311 (+/-0.037)
  ave, std of accuracy of inner CV: 0.417 (+/-0.135)
  ave, std of accuracy of inner CV: 0.388 (+/-0.013)
  ave, std of accuracy of inner CV: 0.498 (+/-0.120)
  ave, std of accuracy of inner CV: 0.412 (+/-0.003)
  ave, std of accuracy of inner CV: 0.350 (+/-0.010)
  ave, std of accuracy of inner CV: 0.222 (+/-0.100)
100.09 seconds 
テスト計算なので、探索範囲を小さくしているせいもあるが、それにしても低い！




[1d1] SVM+OCSVM+ダブルクロスバリデーション/二重交差検証/DCV プログラムをアップデート
ref:
0713, 0716, 0717,0718, 0725
http://univprof.com/archives/16-06-12-3889388.html
https://datachemeng.com/doublecrossvalidation/

* DCVの部分は、どちらでも使えるように作れるので、
  どちらでも使えるバージョンを作って、my_libraryに追加した
  どちらの場合も、modに入れた機械学習方法の.scoreメソッドを用いることになる！
  他の評価方法が欲しければ別だが。
→　それはそれで改造すればいい。


example:
test3_SVM_OCSVM_DCV_rgr.py
test3_SVM_OCSVM_DCV_clf.py
