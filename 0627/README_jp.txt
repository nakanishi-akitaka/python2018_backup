[1b] �T�C�g�ŕ׋��Q�@Python �Ńf�[�^�T�C�G���X
https://pythondatascience.plavox.info/

Pandas �Ńf�[�^�t���[��������
https://pythondatascience.plavox.info/pandas

Pandas �Ńf�[�^�t���[��������Ă݂悤
  test1.py
Pandas �̃f�[�^�t���[�����m�F����
  test2.py
Pandas �Ńf�[�^�t���[���������̍s�E����擾����
  test3.py
Pandas �̃f�[�^�t���[���ɍs��� (�J����) ��ǉ�����
  test4.py
Pandas �̃f�[�^�t���[���̓���̍s�E����폜����
  test5.py
Pandas �̃f�[�^�t���[���̍s�E��̒������m�F����
  test6.py
Pandas �̃f�[�^�t���[���̍s�̗�����ւ���
  test7.py
Pandas �̃f�[�^�t���[�����\�[�g����
  test8.py
Pandas �Ńf�[�^�t���[���̌��� (�}�[�W, JOIN)
  test9.py
Pandas �� CSV �t�@�C����e�L�X�g�t�@�C����ǂݍ���
  test10.py
Pandas �̃f�[�^�t���[���� CSV �t�@�C����e�L�X�g�t�@�C���ɏo�͂���
  test11.py



[1b2] �g���u���V���[�e�B���O
����x�����ł�
C:\Users\Akitaka\Anaconda3\lib\site-packages\spyder
\widgets\variableexplorer\utils.py:414: 
FutureWarning: 'summary' is deprecated and will be removed in a future version.
  display = value.summary()

�Y���t�@�C����ύX
old: display = value.summary()
new: display = value._summary()
ref:
https://github.com/spyder-ide/spyder/issues/7312



[1c] ROC�Ȑ��AAUC
�y�@�B�w�K�z���f���]���E�w�W�ɂ��Ă̂܂Ƃ߂Ǝ��s( w/Titanic�f�[�^�Z�b�g)
https://qiita.com/kenmatsu4/items/0a862a42ceb178ba7155

test12.py
���s����


[1c2]
ROC�Ȑ���AUC�ɂ��Ē�`�Ɗ֌W�����܂Ƃ߂���
https://qiita.com/koyamauchi/items/a2ed9f638b51f3b22cd6
roc_auc_score()�̃T���v�����s���G���[
tet13.py

http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
���Q�l�ɏ�������

[1c3]
�]���w�W���낢��
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
test14.py


ref:04/10
https://mail.google.com/mail/u/0/#sent/LXphbRLrghxkrJntgFpbqNCtZspVcwXcxhpPbMnBxvV

ref:04/11
https://mail.google.com/mail/u/0/#sent/RdDgqcJHpWcvcDjPMnwQkJtsXTSDpJwzPDdxXTbtqTZV

�ǂނׂ��H
http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html


[1c2] �g���u���V���[�e�B���O
��.ix �� .iloc�ɕύX
C:/Users/Akitaka/Downloads/python/0627/test12.py:92: DeprecationWarning: 
.ix is deprecated. Please use
.loc for label based indexing or
.iloc for positional indexing

See the documentation here:
http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated


�t�H�[�}�b�g���C��
old: print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
new: print("Accuracy: {0:.2f} (+/- {1:.2f})".format(scores.mean(), scores.std() * 2))


�x�������
���C�u�����̈ʒu�ύX
* cross_validation
* grid_search

�����ύX
grid_scores_
��
cv_results_
����ƁA���������Ȃ�̂ł���ς���
