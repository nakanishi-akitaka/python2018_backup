[1b] �f�[�^�ϊ�
C:\Users\Akitaka\Downloads\python\20180622\test1.py
tmxo2_gap.csv��ǂݍ���
��CuHO2��[47, 1, 8, 1, 1, 2]�̂悤�ɕϊ�
��test1.csv�ɏ�������

pandas���g���ƁA�ǂ����1�s�ŏI���
�f�[�^�`���ϊ��Ȃǂ͕K�v����

�Q�l
https://pythondatascience.plavox.info/pandas/�f�[�^�t���[�����o�͂���
https://pythondatascience.plavox.info/pandas/csv�t�@�C���̓ǂݍ���



[1c] �M���b�v���@�B�w�K���Ă݂�@��A��
C:\Users\Akitaka\Downloads\python\20180622\test2.py
ref:06/20, 20180620/test1.py���Q�l�ɁA��A��@��������
��RNR�̂݁A�Ȃɂ��G���[�H���N���� 
�@y_pred��NaN���ł�B
��NaN��-1�ɒu������
ref
NumPy�̔z��ndarray�̌����lnp.nan�𑼂̒l�ɒu��
https://note.nkmk.me/python-numpy-nan-replace/

���@
�n�C�p�[�p�����[�^�̓f�t�H���g
X:ABO2�́AA,B�̌��q�ԍ�NA, NB�̂Q�̂�
y:�G�l���M�[�M���b�v
�f�[�^�͑S�� X_train �Ƃ���

����
�X�R�A��RMSE, MAE, RMSE/MAE, R2�̏���
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
�L�͂Ȃ̂́ADTR, RFR, RNR, SVR, GPR



[1d] ���ތv�Z�̃e�X�g
C:\Users\Akitaka\Downloads\python\20180622\test3.py

���̃T���v�����R�s�y���Ă݂�
http://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_probas.html
���W�X�e�B�b�N��A�A�P���x�C�Y���ފ�A�����_���t�H���X�g�A�̂R��
�܂��A�����̑����������s����֐�������


[1b2] �f�[�^�ϊ����̂Q
C:\Users\Akitaka\Downloads\python\20180622\test1.py

�f�[�^��ǂݎ��Ƃ��ɁA
Eg>0 -> 2
Eg=0 -> 1
�ƕϊ����āAtest1_cls.csv�ɏ�������
 

[1e] �M���b�v���@�B�w�K���Ă݂�@���ޕ�
Eg�̐��l���u��A�v�ł͂Ȃ��A�������≏�̂����u���ށv����

���@
����؂̂�
�n�C�p�[�p�����[�^�̓f�t�H���g
X:ABO2�́AA,B�̌��q�ԍ�NA, NB�̂Q�̂�
y:�≏��=1, ����=2
�f�[�^�͑S�� X_train �Ƃ���B���؁E�\���Ȃ�

�]�����@��
python_work_fs01/2018/0406/test2.py���Q�l�ɐF�X����

����
metrics.confusion_matrix
[[26  0]
 [ 0 70]]
���S�ɕ��ނł���
�ߊw�K�̉\��������̂ŁA��T�Ɋ�ׂȂ���


[1f] �_���ǂ�
ref:06/05 <div>
���q������twitter���
https://twitter.com/hirokaneko226/status/1002123554463272960
���������y�f�Ǝ_���Ҍ��y�f�Ɋւ����Ӕ������N���X���ނ���_���B

Multiclassification Prediction of Enzymatic Reactions
 for Oxidoreductases and Hydrolases
 Using Reaction Fingerprints and Machine Learning Methods
Yingchun Cai et al. (China group)
J. Chem. Inf. Model.
DOI: 10.1021/acs.jcim.7b00656
�����A�y�[�W�Ȃǂ͖���B�d�q�ł̂݌��J���H

�}�P�ɂ��ƁA
 �������������{����������Ȃ遨���ꂼ��ɂ���AP,Mogan2,TT,PF������
 ������4����RDF,SRF,TRF���v�Z�H������3�𔽉��t�B���K�[�v�����g�Ƃ���
 KEGG�������t�B���K�[�v�����g�̃f�[�^���聨training��test�ɕ�����
 Rhea(�f�[�^�x�[�X)�������`�̃f�[�^���聨validation�ɂ���
 training����@�B�w�K�Ń��f���쐬
 �`���[�j���O����better���f����
 test��optimal���f����
 optimal���f����validation�Ō���

�N���X���ގ�@�͂V��ނŌ����B
��decision tree (DT), k-nearest neighbors (k-NN), logistic regression (LR),
 naive Bayes (NB), neural network (NN), random forest (RF), and
support vector machine (SVM)

transformation reaction fingerprint �Ń��W�X�e�B�b�N��A�E�j���[�����l�b�g�̃P�[�X�����\���ǂ������B
</div>
