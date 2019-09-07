# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 10:31:42 2018

@author: Akitaka
"""




[1c1] OCSVMもモジュール化
モジュール化は完了
example:
test0_SVM_OCSVM_DCV_clf.py


[1c2] kNNとOCSVMのADを比べてみる
等高線の図に、+かxで表示することで、確認
    kNNではAD外だがOCSVMではAD内、あるいはその逆、であるサンプルの数はあまり変わらない
計算がうまくいったかも確認
    見た限りでは特におかしい所はない

example:
test0_SVM_OCSVM_kNN_clf_contour.py
ref:
summary/0726test0_SVM_OCSVM_contour.py
http://ailaby.com/plot_marker/

ref:
モジュール化について検索
https://www.sejuku.net/blog/25587
http://maku77.github.io/python/env/create-module.html
https://qiita.com/Tocyuki/items/fb99d9bdb71875843357
https://qiita.com/Usek/items/86edfa0835292c80fff5




[1c3] OCSVMモジュール化をTc予測に実装
example:
test2_Tc_SVM_OCSVM_DCV.py




[1d1] RFでTc予測
ADはkNNを使用
予測結果は以下の通り。
RFの仕様上、トレーニングデータのTcを超えることはない。

formula,P,Tc,AD
H3Cl,200,196,1
H3S,200,196,1
H3S,250,188,1
H3Cl,250,188,1
H3S2,200,185,1
H3Cl2,200,185,1
TiH3,200,180,1
VH3,200,180,1
ScH3,200,180,1
KH3,200,180,1

example:
test2_Tc_SVM_AD_DCV.py




[1e1] データベースのRef元の論文をDL
    「～の組成と圧力では安定でない」という情報もデータベースにする
ref:
/work/share/superconductivity_20180109.data
ここのRefはすべてDL完了！

自分の論文のTable 1
12) F. Peng, Phys. Rev. Lett 119, 107001 (2017).
13) H. Liu, Proc. Natl. Acad. Sci. U.S.A. 114, 6990-6995 (2017).
14) X. Feng, RSC Adv. 5, 59292-59296 (2015).
15) Y. Li, Sci. Rep. 5, 9948 (2015).
16) H. Wang, Proc. Natl Acad.Sci. U.S.A. 109, 6463 (2012). [DB]
17) J. Feng, Phys. Rev. Lett. 96, 017006 (2006).
18) Y. Fu, Chem. Mater., 28 (6), pp 1746-1755 (2016). [DB]
19) P. Hou, RSC Adv. 5, 5096-5101 (2015). [DB]
20) Y. Ma,  arXiv:1511.05291. [DB]
21) Y. Ma,  arXiv:1506.03889.
22) Y. Li, Phys. Rev.B 82, 064104 (2010).
23) Y. Cheng, Sci. Rep. 5, 16475 (2015).
24) T. Ishikawa, J. Phys. Soc. Jpn. 86, 124711 (2017). [DB]

More information　is summarized in Refs.12, 25–28)
25) D, Duan, National Science　Review 4, 121 (2017).
26) H.Wang, WIREs Comput. Mol. Sci. 8,　e1330 (2018).
27) E. Zurek, Comments Inorg. Chem. 37, 78 (2017). [有料]
28) L. Zhang, Nature Reviews Materials 2, 17005　(2017).

＋
たまたま見つけたやつ Ru-H
Y. Liu, Phys. Chem. Chem. Phys., 2016, 18, 1516-1520
