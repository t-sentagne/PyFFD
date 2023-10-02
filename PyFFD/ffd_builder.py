# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 10:55:32 2023

@author: SENTAGNE
"""
import numpy as np
from .nurbs import Get2dBasisFunctionsAtPts, Get3dBasisFunctionsAtPts
import scipy.sparse as sps


def PreConditioning(N):
    N_tilde = N@N.T
    D = sps.diags(1/np.sqrt(N_tilde.diagonal()))
    N = D @ N
    return N


class FFD:
    def __init__(self, n_fem, ne_s, deg_s, **kwargs):
        """
        B-Spline box generation class.

        Parameters
        ----------
        n_fem : 2D array
            Coordonate of node for each directions.
        ne_s : 1D array
            Number of spline element for each directions.
        deg_s : int
            Degree value (p).
        **kwargs : TYPE
            DESCRIPTION.

        """
        self.dim = n_fem.shape[1]

        self.n_fem = n_fem
        self.ne_s = ne_s
        if self.dim != len(ne_s):
            raise NameError("Size of 'ne_s' not correct.")
        self.deg_s = deg_s
        self.delta = kwargs.pop('delta', 0)

        self.knot_vectors = []
        self.crtpts_del = []

        self.KnotVectorBuilder()

    def KnotVectorBuilder(self):
        for n in range(self.dim):
            knot_vector_init = self.KnotVectorInit(self.ne_s[n]+1)
            knot_vector_scaled = self.KnotVectorScaling(
                self.n_fem[:, n], knot_vector_init)  # FIXME: maybe [] needed
            self.knot_vectors.append(knot_vector_scaled)
            print("Knot vectors prepared")

    def KnotVectorInit(self, ndof):
        return np.concatenate((np.zeros(self.deg_s), np.linspace(0, 1, ndof),
                               np.ones(self.deg_s)))

    def KnotVectorScaling(self, coord, knot):
        return (max(coord)-min(coord)+self.delta) * knot + min(coord) - self.delta/2

    def OperatorFFD(self, precond=True, limit=1e-10):
        self.limit = limit
        if self.dim == 2:
            N = Get2dBasisFunctionsAtPts(self.n_fem[:, 0], self.n_fem[:, 1], self.knot_vectors[0],
                                         self.knot_vectors[1], self.deg_s, self.deg_s)
        elif self.dim == 3:
            N = Get3dBasisFunctionsAtPts(self.n_fem[:, 0], self.n_fem[:, 1], self.n_fem[:, 2],
                                         self.knot_vectors[0], self.knot_vectors[1], self.knot_vectors[2],
                                         self.deg_s, self.deg_s, self.deg_s)

        else:
            raise NameError('Dimension not valid to build the operator.')

        N_reduct = self.ClearControlPoint(N)

        if precond == True:
            print("Rs preconditioning")
            N_reduct = PreConditioning(N_reduct)
        return N_reduct

    def ClearControlPoint(self, N):
        sum_N = N.sum(axis=1)

        idc, _ = np.where(sum_N > self.limit)
        N_reduct = N[idc]

        val = np.arange(len(sum_N))
        self.idc_del = np.delete(val, idc)
        return N_reduct
