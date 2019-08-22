#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:set foldmethod=marker:

#
# convert tc.csv
#
#   S2H5  ,           1,   9.88167642857, 0.71428571428,   100  , 1.0786,  902.09 ,0.17,   49.5486   
#   S2H5  ,           1,   9.88167642857, 0.71428571428,   112  , 1.1857,  897.9  ,0.17,   58.3099   
#   S2H5  ,           1,   9.88167642857, 0.71428571428,   120  , 1.2390,  907.0  ,0.17,   63.2085   
#   S2H5  ,           1,   9.88167642857, 0.71428571428,   130  , 1.2417,  950.7  ,0.17,   66.4768   
#   S2H5  ,           1,   9.88167642857, 0.71428571428,   140  , 1.2597,  968.93 ,0.17,   69.2659   
# -> 
#   49.5486, 16.0, 1.0, 2.0, 5.0, 100.0
#   58.3099, 16.0, 1.0, 2.0, 5.0, 112.0
#   63.2085, 16.0, 1.0, 2.0, 5.0, 120.0
#   66.4768, 16.0, 1.0, 2.0, 5.0, 130.0
#   69.2659, 16.0, 1.0, 2.0, 5.0, 140.0

# modules
# {{{
import numpy as np
from time import time
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston
from pymatgen import Composition, Element
from numpy import zeros, mean
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import LeaveOneOut
from sklearn import linear_model, metrics, ensemble
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
# }}}
#
# read from tc.csv
# {{{
trainFile = open("tc.csv","r").readlines()
tc = []
pf = []
yx = []
for line in trainFile:
    split = str.split(line, ',')
    material = Composition(split[0])
    pressure = float(split[4])
    tc.append(float(split[8]))
    features = []
    atomicNo = []
    natom = []

    for element in material:
        natom.append(material.get_atomic_fraction(element)*material.num_atoms)
        atomicNo.append(float(element.Z))

    features.append(float(split[8]))
    features.extend(atomicNo)
    features.extend(natom)
    features.append(pressure)
    yx.append(features[:])
# }}}
#
# set X_train, y_train, X_test
# {{{
X = pf[:]
y = tc[:]
X_train = X[:]
y_train = y[:]


file = open('tc_train.csv','w')
for i in range(len(yx)):
#   print(yx[i])
#   print('{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.\
#   format(yx[i][0], yx[i][1], yx[i][2], yx[i][3], yx[i][4], yx[i][5]))
#   print('{0[0]}, {0[1]}, {0[2]}, {0[3]}, {0[4]}, {0[5]}'.format(yx[i]))
    file.write('{0[0]}, {0[1]}, {0[2]}, {0[3]}, {0[4]}, {0[5]}\n'.format(yx[i]))
file.close()

materials = []
xatom=Element("H")
for i in range(3,86):
    if(not xatom.from_Z(i).is_noble_gas):
        for iatom1 in range(1,10):
            for iatom2 in range(1,10):
#               print('%s%.1i%s%.1i' % (xatom.from_Z(i).symbol,iatom1,xatom.symbol,iatom2))
                str_mat=str(xatom.from_Z(i).symbol)+str(iatom1)+str(xatom.symbol)+str(iatom2)
                materials.append(Composition(str_mat))

X_test = []
for material in materials:
    atomicNo = []
    natom = []
    for element in material:
        natom.append(material.get_atomic_fraction(element)*material.num_atoms)
        atomicNo.append(float(element.Z))
#   atom0=element.from_Z(atomicNo[0]).symbol
#   atom1=element.from_Z(atomicNo[1]).symbol
#   print('%s%.1i %s%.1i' % (atom0,natom[0],atom1,natom[1]))
#   print('%s%.1i %s%.1i' % (element.from_Z(atomicNo[0]).symbol,natom[0], \
#                            element.from_Z(atomicNo[1]).symbol,natom[1]))
    for ip in range( 50,550,50):
        temp = []
        temp.append(0) # dummy tc
        temp.extend(atomicNo)
        temp.extend(natom)
        temp.append(float(ip))
        X_test.append(temp[:])
#       print(temp)
file = open('tc_test.csv','w')
for i in range(len(X_test)):
    file.write('{0[0]}, {0[1]}, {0[2]}, {0[3]}, {0[4]}, {0[5]}\n'.format(X_test[i]))
file.close()
# }}}
