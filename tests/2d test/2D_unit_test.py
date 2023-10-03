# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 18:00:25 2023

@author: SENTAGNE
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

import PyFFD as pf
import numpy as np
import scipy.sparse as sps
import meshio


mesh = meshio.read("mesh_test.msh")
n = mesh.points[:,:2]
Rs_theoric = sps.load_npz("Rs_unit_test.npz")


deg_s = 3
ne_s = np.array([10, 5])
delta = 0.1

ms = pf.FFD(n, ne_s, deg_s, delta=delta)
Rs = ms.OperatorFFD()

# %% Test
residu = Rs - Rs_theoric

if residu.A.all() == 0:
    print("\nLibrary verified")
else:
    print("\nResultat is wrong")
