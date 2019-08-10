#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:set foldmethod=marker:

#
# convert tc.csv
#
#   S2H5, 1,   9.88167642857, 0.71428571428,   100  , 1.0786,  902.09 ,0.17,   49.5486   
#   S2H5, 1,   9.88167642857, 0.71428571428,   112  , 1.1857,  897.9  ,0.17,   58.3099   
#   S2H5, 1,   9.88167642857, 0.71428571428,   120  , 1.2390,  907.0  ,0.17,   63.2085   
#   S2H5, 1,   9.88167642857, 0.71428571428,   130  , 1.2417,  950.7  ,0.17,   66.4768   
#   S2H5, 1,   9.88167642857, 0.71428571428,   140  , 1.2597,  968.93 ,0.17,   69.2659   
# -> 
#   49.5486, 16.0, 1.0, 2.0, 5.0, 100.0
#   58.3099, 16.0, 1.0, 2.0, 5.0, 112.0
#   63.2085, 16.0, 1.0, 2.0, 5.0, 120.0
#   66.4768, 16.0, 1.0, 2.0, 5.0, 130.0
#   69.2659, 16.0, 1.0, 2.0, 5.0, 140.0

# modules
# {{{
from pymatgen import Composition
# }}}
#
# read from tc.csv
# {{{
trainFile = open("tmxo2.csv","r").readlines()
eg = []
yx = []
for line in trainFile:
    print(line)
    split = str.split(line, ',')
    material = Composition(split[0])
    eg.append(float(split[3]))
    features = []
    atomicNo = []
    natom = []
    for element in material:
        natom.append(material.get_atomic_fraction(element)*material.num_atoms)
        atomicNo.append(float(element.Z))
    features.append(float(split[3]))
    features.extend(atomicNo)
    features.extend(natom)
    yx.append(features[:])
# }}}
#
# set X_train, y_train, X_test
# {{{

file = open('tmxo2_train.csv','w')
for i in range(len(yx)):
#   print(yx[i])
#   print('{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.\
#   format(yx[i][0], yx[i][1], yx[i][2], yx[i][3], yx[i][4], yx[i][5]))
#   print('{0[0]}, {0[1]}, {0[2]}, {0[3]}, {0[4]}, {0[5]}'.format(yx[i]))
    file.write('{0[0]}, {0[1]}, {0[2]}, {0[3]}, {0[4]}\n'.format(yx[i]))
file.close()
# }}}
