# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 13:49:57 2018

@author: Akitaka
"""
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

import numpy as np
import pandas as pd
from pymatgen import Composition
from pymatgen import periodic_table 
#
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

yx = np.array(yx)
y_train = yx[:,0]
X_train = yx[:,1:]

ltest = False
if(ltest):
    X = X_train
    print(X[0][:4])
    atom1=periodic_table.get_el_sp(X[0][0])
    atom2=periodic_table.get_el_sp(X[0][1])
    str_mat=str(atom1)+str(X[0][2])+str(atom2)+str(X[0][3])
    material = Composition(str_mat)
    print(type(material.reduced_formula))
    for i in range(1,2):
        print(periodic_table.get_el_sp(i))

properties=['Tc','Z1','Z2','N1', 'N2', 'P' ]
df = pd.DataFrame(yx, columns=properties)
df.to_csv("tc_train.csv")

# list of A_n H_m (n,m=1,...,10)
tc = 0.0
yx = []
zatom2 = 1
atom2 = periodic_table.get_el_sp(zatom2)
for zatom1 in range(3,86):
    atom1 = periodic_table.get_el_sp(zatom1)
    if(not atom1.is_noble_gas):
        for natom1 in range(1,10):
            for natom2 in range(1,10):
                for ip in range( 50,550,50):
                    temp = [tc, zatom1, zatom2, natom1, natom2, ip]
                    yx.append(temp[:])

properties=['Tc','Z1','Z2','N1', 'N2', 'P' ]
df = pd.DataFrame(yx, columns=properties)
df.to_csv("tc_test.csv")