# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 21:46:54 2018

@author: Akitaka
http://pymatgen.org/

"""

# Quick start
import pymatgen as mg
si = mg.Element("Si")
print(si.atomic_mass)
print(si.melting_point)

comp = mg.Composition("Fe2O3")
print(comp.weight)

# Note that Composition conveniently allows strings to be treated just
# like an Element object.
print(comp["Fe"])
print(comp.get_atomic_fraction("Fe"))
lattice = mg.Lattice.cubic(4.2)
structure = mg.Structure(lattice, ["Cs", "Cl"],
                         [[0, 0, 0], [0.5, 0.5, 0.5]])
print(structure.volume)
print(structure[0])
# You can create a Structure using spacegroup symmetry as well.
li2o = mg.Structure.from_spacegroup("Fm-3m", mg.Lattice.cubic(3),
       ["Li", "O"], [[0.25, 0.25, 0.25], [0, 0, 0]])

# Integrated symmetry analysis tools from spglib.
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
finder = SpacegroupAnalyzer(structure)
# print(finder.get_spacegroup_symbol())

# Convenient IO to various formats. You can specify various formats.
# Without a filename, a string is returned. Otherwise,
# the output is written to the file. If only the filenmae is provided,
# the format is intelligently determined from a file.
# structure.to(fmt="poscar")
# structure.to(filename="POSCAR")
# structure.to(filename="CsCl.cif")

# Pythonic API for editing Structures and Molecules (v2.9.1 onwards)
# Changing the specie of a site.
structure[1] = "F"
print(structure)

# Changes species and coordinates (fractional assumed for structures)    
structure[1] = "Cl", [0.51, 0.51, 0.51]
print(structure)

# Replaces all Cs in the structure with K
structure["Cs"] = "K"
print(structure)

# Replaces all K in the structure with K: 0.5, Na: 0.5, i.e.,
# a disordered structure is created.
structure["K"] = "K0.5Na0.5"
print(structure)

# Because structure is like a list, it supports most list-like methods
# such as sort, reverse, etc.
structure.reverse()
print(structure)
