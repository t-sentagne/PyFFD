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


def KnotVectorInit(ndof, deg):
    return np.concatenate((np.zeros(deg), np.linspace(0, 1, ndof),
                           np.ones(deg)))


def KnotVectorScaling(coord_min, coord_max, knot):
    return (coord_max-coord_min) * knot + coord_min


class FFD:
    def __init__(self, n, size, deg, **kwargs):
        """
        B-Spline box generation class.

        Parameters
        ----------
        n : 2D array
            Coordinate of node in each directions.
        ne_s : 1D array
            Number of spline element in each directions.
        deg : int
            Degree value (p).
        **kwargs : TYPE
            DESCRIPTION.

        """
        self.n = n
        self.dim = n.shape[1]
        if type(deg) == int:
            deg = [deg,] * self.dim
        self.deg = deg
        if type(size) == float:
            size = [size,] * self.dim
            print("size =", size)
        self.size = size

        self.crtpts_del = []
        self.limit = 1e-10
        self.precond = True
        self.SetMarginThickness()

    def SetThreshold(self, limit):
        self.limit = limit

    def SetPreconditioning(self, precond):
        if type(precond) != bool:
            raise NameError("Boolean is expected for 'precond' parameter")
        self.precond = precond

    def SetMarginThickness(self, delta='tight'):
        """

        Parameters
        ----------
        delta : STRING
            'tight'
            'value'
            'uniform'

        Returns
        -------
        None.

        """
        if type(delta) == float:
            val_delta = [delta,]*self.dim

        elif delta == 'fitted':
            val_delta = np.array([0,]*self.dim)  # FIXME: more complex

        elif delta == 'tight':
            val_delta = np.array(self.size)*2  # FIXME: more complex

        else:
            val_delta = delta

        self.delta = np.array(val_delta)
        self.KnotVectorBuilder()

    def KnotVectorBuilder(self):
        self.knot_vectors = []
        self.ne = []
        for i in range(self.dim):
            coord_min = self.n[:, i].min() - self.delta[i]/2
            coord_max = self.n[:, i].max() + self.delta[i]/2
            length = coord_max - coord_min
            n_el = np.round(length/self.size[i]).astype('int')
            self.ne.append(n_el)

            knot_vector_init = KnotVectorInit(n_el+1, self.deg[i])
            knot_vector_scaled = KnotVectorScaling(
                coord_min, coord_max, knot_vector_init)
            self.knot_vectors.append(knot_vector_scaled)

    def OperatorFFD(self):
        if self.dim == 2:
            N = Get2dBasisFunctionsAtPts(self.n[:, 0], self.n[:, 1],
                                         self.knot_vectors[0], self.knot_vectors[1],
                                         self.deg[0], self.deg[1])
        elif self.dim == 3:
            N = Get3dBasisFunctionsAtPts(self.n[:, 0], self.n[:, 1], self.n[:, 2],
                                         self.knot_vectors[0], self.knot_vectors[1], self.knot_vectors[2],
                                         self.deg[0], self.deg[1], self.deg[2])

        N_reduct = self.ClearControlPoint(N)

        if self.precond:
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
