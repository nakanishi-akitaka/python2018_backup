# -*- coding: utf-8 -*-
"""
(1) convert tc.csv -> tc_train.csv
chemical formula -> atomic No. & number of atoms 
 ex) S2H5  -> 16.0, 1.0, 2.0, 5.0

(2) make tc_test.csv
   all A_n H_m (n,m=1,...,10)

Created on Thu Aug  2 15:41:37 2018

@author: Akitaka
"""



import numpy as np
import pandas as pd
from pymatgen import Composition
from pymatgen import periodic_table 
#
trainFile = open("tc.csv","r").readlines()
yx = []
for line in trainFile:
    split = str.split(line, ',')
    material = Composition(split[0])
    pressure = float(split[4])
    tc = float(split[8])
    features = []
    atomicNo = []
    natom = []

    for element in material:
        natom.append(material.get_atomic_fraction(element)*material.num_atoms)
        atomicNo.append(float(element.Z))

    features.append(tc)
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

