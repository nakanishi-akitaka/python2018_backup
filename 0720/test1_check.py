# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 15:13:17 2018

@author: Akitaka
"""

from pymatgen import Composition

trainFile = open("tc.csv","r").readlines()
tc        = []
pf = []

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

    features.extend(atomicNo)
    features.extend(natom)
    features.append(pressure)
    pf.append(features[:])
X = pf[:]
y = tc[:]
import numpy as np
import pandas as pd
Z = np.array(X)

def read_file(name): 
    data = np.array(pd.read_csv(filepath_or_buffer=name,
                               header=None,sep=','))[:,:]
    y = data[:,0]
    X = data[:,1:]
    return X, y

#
# start of machine learning
#

train_file = 'tc_train.csv'
X, y = read_file(train_file)
for i in range(len(X)):
    if(X[i].all()==Z[i].all()):
        print('OK')
    else:
        print('NG')