# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 18:00:25 2023

@author: SENTAGNE
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

import src.PyFFD as pf
import numpy as np
import scipy.sparse as sps
import meshio


mesh = meshio.read("mesh_test.msh")
n = mesh.points[:,:2]
Rs_theoric = sps.load_npz("Rs_unit_test.npz")


deg_s = 3



ms = pf.FFD(n, 0.2, 3)
ms.SetMarginThickness(0.1)
Rs = ms.OperatorFFD()

# %% Test
residu = Rs - Rs_theoric

if (residu.A <= 1e-14).all():
    print("\nLibrary verified")
else:
    print("\nResultat is wrong !")
