scikit-learn �ŋ@�B�w�K
https://pythondatascience.plavox.info/scikit-learn

scikit-learn �Ńg���[�j���O�f�[�^�ƃe�X�g�f�[�^���쐬����
  test1.py
scikit-learn �Ő��`��A (�P��A���́E�d��A����)
  test2.py
scikit-learn �ŃN���X�^���� (K-means �@)
  test3.py
scikit-learn �Ō���ؕ��� (CART �@)
  test4.py
scikit-learn �ŃN���X���ތ��ʂ�]������
  test5.py
scikit-learn �ŉ�A���f���̌��ʂ�]������
  test6.py
scikit-learn �ɕt�����Ă���f�[�^�Z�b�g
  test7.py

����؃O���t����}���邽�߁AGraphviz���C���X�g�[��
https://graphviz.gitlab.io/download/

pydotplus��pip�ŃC���X�g�[��
https://pypi.python.org/pypi/pydotplus



[1c] ���ނ̕]���w�W�ɂ���
�����s��@wikipedia
https://en.wikipedia.org/wiki/Confusion_matrix
�F��ȃp�^�[��������I

�@�B�w�K�Ŏg���w�W���܂Ƃ�(���t����w�K��)
http://www.procrasist.com/entry/ml-metrics
�ǂ̕]�����g�����́A�ړI�ɂ��ƌ�����
��1:�ʕ����݂���orNot�ŕ��ނ���ꍇ�́A
�@���������ނł��Ă��邩�ǂ������厖�Ȃ̂ŁA����/Accuracy
��2:���񌟐f�́A�K�������N�ƌ�f����̂���Ԕ�������
�@�t�Ɍ��N���K���Ɛf�f���Ă��\��Ȃ��i�K���Ɛf�f�����ꍇ�͂ǂ��ł������j
�@�̂ŁA�Č���/Recall���P�ɂ��邱�Ƃ��厖
�܂Ƃ�
* �����臒l��ݒ肹���ɕ]���������ꍇ -> ROC, AUC
* �����臒l��ݒ肵�ĕ]���������ꍇ
** A�ł���AA�łȂ��̏d�v�x���ꏏ�̏ꍇ Accuracy
** A�ł���ƌ����������d�v�ȏꍇ
*** A�ł���Ɨ\�����āA���ۂ�A�ł������l�̊������d�v
    (�����A�Č����ɃR�X�g��������Ȃ�)�ȏꍇ -> Precision
*** ���ۂ�A�ł���l���������A�Ɨ\���ł��Ă��邩(Recall)��
    �d�v(���m�R�ꂪ������Ȃ�) -> Recall
*** �������������ꍇ -> (weighted) F-measure



[1d] �_���ł͂ǂ�ȕ]���w�W���g���Ă��邩�H���ׂ�
"Prediction of Low-Thermal-Conductivity Compounds with
 First-Principles Anharmonic Lattice-Dynamics Calculations and Bayesian Optimization"
Atsuto Seko, Atsushi Togo, Hiroyuki Hayashi, Koji Tsuda, Laurent Chaput, and Isao Tanaka
Phys. Rev. Lett. 115, 205901 - Published 10 November 2015
DOI:https://doi.org/10.1103/PhysRevLett.115.205901
�s��

"Multiclassification Prediction of Enzymatic Reactions for Oxidoreductases and Hydrolases
 Using Reaction Fingerprints and Machine Learning Methods"
Yingchun Cai , Hongbin Yang , Weihua Li , Guixia Liu, Philip W. Lee, and Yun Tang* 
J. Chem. Inf. Model., 2018, 58 (6), pp 1169-1181
DOI: 10.1021/acs.jcim.7b00656

���f���̕]�� = Precision, Recall, F1-score
p1170
> In addition, three metrics P (Precision), R (Recall), and F (F1-score)
>  were used to evaluate the performance of each model


"Machine learning modeling of superconducting critical temperature"
https://arxiv.org/abs/1709.02727
���`���ł̋@�B�w�K���A���łɏo���B
�H�@���̂�邱�Ƃ͉�������H
12000+SuperCon�f�[�^�x�[�X
1.�g���݂̂�����ʂƂ��āATc 10K�ȏ�Ɩ�����2�ʂ�ɕ����郂�f�����\�z�@����92%
2.�����Tc����̓I�ɗ\�����郂�f���ɉ���
�ق��̓����ʂ�AFLOW����擾���邱�Ƃł���ɉ���
��30�̌��(�񓺎_�����A��S�_����)�𓾂���Table 3
�@Tc > 20K������̓I�ɂ͕s��
�����_���t�H���X�g��A
ref:2017/09/11

���f���̕]�� = Accuracy, Precision, Recall, F1-score
p4
> Hypothetically, if 95% of the observations in the
> dataset are in the below-Tsep group, simply classifying
> all materials as such would yield a high accuracy (95%),
> while being trivial in any other sense. To avoid this potential
> pitfall, three other standard metrics for classification
> are considered: precision, recall, and F1 score. They
> are defined using the values tp, tn, f p, and fn for the
> count of true/false positive/negative predictions of the model:


��̘_�������p�����_��
https://www.researchgate.net/publication/319622538_Machine_learning_modeling_of_superconducting_critical_temperature
"A Data-Driven Statistical Model for Predicting the Critical Temperature of a Superconductor"
https://arxiv.org/abs/1803.10260

�g�p�������������W��
 = Atomic Mass, First Ionization Energy, Atomic Radius, Density,
   Electron Affinity, Fusion Heat, Thermal Conductivity, Valence  
Table 1: This table shows the properties of an element which are used for creating features to
predict Tc.

������쐬����������
��FThermal Conductivity �̕��ρA�d�ݕt�����ρA���敽�ςȂ�
Table 2: This table summarizes the procedure for feature extraction from material�fs chemical formula

�]���w�W = RMSE, R^2
> Our XGBoost model gives good predictions: an out-of-sample error of about 9.5 K
> based on root-mean-squared-error (rmse), and an out-of-sample R2 values of about 0.92. T


�}�e���A���E�C���t�H�}�e�B�N�X�_���ǂ݁{�c�_
"Accelerated Materials Design of Lithium Superionic Conductors Based
on First-Principles Calculations and Machine Learning Algorithms"
Koji Fujimura
Adv Energy Mater 3. 980 (2013)
DOI: 10.1002/aenm.201300060
ref:2016/12/06

�]���w�W ???
> The variance of the Gaussian kernel, the regularization
> constant and forms of independent variables were optimized
> by minimizing the prediction error estimated by the bootstrap-
> ping method. [36] The prediction error of the optimized SVR for
> log �� is 0.373. 


������\���ɂ��Ă̋@�B�w�K
"The thermodynamic scale of inorganic crystalline metastability"
Wenhao Sun et al., Science Advances  18 Nov 2016:Vol. 2, no. 11, e1600225
DOI: 10.1126/sciadv.1600225


