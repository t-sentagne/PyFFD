# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 18:00:25 2023

@author: SENTAGNE
"""
import os
import sys
import PyFFD as pf
import numpy as np
import scipy.sparse as sps
import meshio
import matplotlib.pyplot as plt

mesh = meshio.read("mesh_test.msh")
n = mesh.points[:, :2]
Rs_theoric = sps.load_npz("Rs_unit_test.npz")


ms = pf.FFD(n, 0.2, 4)
ms.SetMarginThickness(0.)

Rs = ms.OperatorFFD()
idc_bc = ms.SelectNodesBox()

bc = np.zeros((ms.nc,2))
bc[idc_bc] += 0.5

ms.Morphing(bc)
ms.PlotPoint()


# %% Test
# residu = Rs - Rs_theoric

# if (residu.A <= 1e-14).all():
#     print("\nLibrary verified")
# else:
#     print("\nResultat is wrong !")





