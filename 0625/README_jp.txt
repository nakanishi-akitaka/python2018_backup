[1c] t-SNE�̊ȒP�ȃT���v�������s
C:\Users\Akitaka\Downloads\python\20180623/test1.ipynb PCA
C:\Users\Akitaka\Downloads\python\20180623/test2.ipynb t-SNE

scikit-learn��t-SNE�U�z�}��`���Ă݂�
http://inaz2.hatenablog.com/entry/2017/01/24/211331
PCA�Ɣ�r�������́B���͗�R

Python: ���l�̊w�K (Manifold Learning) ��p���������k��
https://blog.amedama.jp/entry/2017/12/09/142655
C:\Users\Akitaka\Downloads\python\20180623/test3.ipynb digit�̃e�X�g
C:\Users\Akitaka\Downloads\python\20180623/test4.ipynb ��r
PCA, MDS, Isomap, LocallyLinearEmbedding, Laplacian Eigenmaps �Ɣ�r�������́B���͗�R


[1d] �M���b�v���@�B�w�K���Ă݂�@���ޕ�
Eg�̐��l���u��A�v�ł͂Ȃ��A�������≏�̂����u���ށv����
ref:06/22

C:\Users\Akitaka\Downloads\python\20180625\test5.py
���@
�n�C�p�[�p�����[�^�̓f�t�H���g
X:ABO2�́AA,B�̌��q�ԍ�NA, NB�̂Q�̂�
y:����(1)���≏��(2)��
�f�[�^�͑S�� X_train �Ƃ���
��@�͈ȉ��̂Q�P�S�����s
[���s�p�v���O�������J] �Q�P�̔��ʕ���(��N���X����)��@����C�Ɏ��s���Č��ʂ��r����I
�_�u���N���X�o���f�[�V�����ɂ��]���t�� (Python����)
https://note.mu/univprof/n/n38855bb9bfa8

����
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

��18. LSVM�Ɋ�Â�Adaptive Boosting (AdaBoost[LSVM])�̂݁A�G���[���o��
ValueError: BaseClassifier in AdaBoostClassifier ensemble
 is worse than random, ensemble can not be fit.
���18.�̒l��SVL�̂܂�

�S���≏�̂ɕ��ނ������̂�����B����ł����X�≏�̂������Ɛ��x���オ���Ă��܂��I
��AUC�AROC�Ȑ����g���ׂ��ł� ref:04/10-11