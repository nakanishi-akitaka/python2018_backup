# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 11:12:16 2018

@author: Akitaka
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:set foldmethod=marker:
# convert Chemical formula -> atomic number & number of atoms 
# modules
from pymatgen import Composition
import numpy as np
import pandas as pd
#
# read from csv
# {{{
name = 'tmxo2_gap.csv'
data = np.array(pd.read_csv(name))[:,:]
y=data[:,0]
X=data[:,1:]

trainFile = open("tmxo2_gap.csv","r").readlines()
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
for i in range(len(yx)):
    print('{0[0]}, {0[1]}, {0[2]}, {0[3]}, {0[4]}, {0[5]}, {0[6]}\n'.format(yx[i]))
# }}}