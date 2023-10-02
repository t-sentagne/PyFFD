# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 18:00:25 2023

@author: SENTAGNE
"""

import PyFFD.PyFFD as pf
import numpy as np
import scipy.sparse as sps

n_compress = np.load("FEM_mesh.npz")
n = n_compress['n']  # Loading of FEM mesh
Rs_theoric = sps.load_npz("Rs.npz")


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
