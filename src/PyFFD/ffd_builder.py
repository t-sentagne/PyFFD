# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 10:55:32 2023

@author: SENTAGNE
"""
import numpy as np
from .nurbs import Get2dBasisFunctionsAtPts, Get3dBasisFunctionsAtPts
import scipy.sparse as sps
import matplotlib.pyplot as plt
from .vtktools import VTRWriter


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


def GrevilleAbscissaes(knot_vect, deg):
    """
    Compute the Greville abscissaes to return the control points coordinates.

    Parameters
    ----------
    knot_vect : np.array
        The knot vector
    deg : int
        Degree of spline

    Returns
    -------
    x : np.array
        Control points coordinates

    """
    m = len(knot_vect)
    x = np.zeros(m-deg-1)
    for k in range(1, m-deg):
        x[k-1] = 1/deg * np.sum(knot_vect[k:k+deg])
    return x


def MeshGrid(vect):
    if len(vect) == 2:
        coord = np.meshgrid(vect[0], vect[1])
    else:
        coord = np.meshgrid(vect[0], vect[1], vect[2])

    coord_ravel = []
    for nb in range(len(coord)):
        coord_ravel.append(np.ravel(coord[nb]))

    coord_col = np.column_stack(coord_ravel)
    return coord_col


class FFD:
    def __init__(self, npts, size, deg, **kwargs):
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
        self.npts = npts
        self.dim = npts.shape[1]
        if type(deg) == int:
            deg = [deg,] * self.dim
        self.deg = deg
        if type(size) != list:
            size = [size,] * self.dim
        self.size = size

        self.e = []
        self.c = []

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
        if type(delta) != list and type(delta) != str:
            val_delta = [delta,]*self.dim

        elif delta == 'fitted':
            val_delta = np.array([0,]*self.dim)

        elif delta == 'tight':
            val_delta = np.array(self.size)*2

        else:
            val_delta = delta

        self.delta = np.array(val_delta)
        self.KnotVectorBuilder()

    def KnotVectorBuilder(self):
        self.knot_vectors = []
        self.ne = []
        ctrlpts_posi = []
        node_posi = []

        for i in range(self.dim):
            coord_min = self.npts[:, i].min() - self.delta[i]/2
            coord_max = self.npts[:, i].max() + self.delta[i]/2
            length = coord_max - coord_min
            n_el = np.round(length/self.size[i]).astype('int')
            self.ne.append(n_el)

            knot_vector_init = KnotVectorInit(n_el+1, self.deg[i])
            knot_vector_scaled = KnotVectorScaling(
                coord_min, coord_max, knot_vector_init)

            ctrlpts_posi.append(GrevilleAbscissaes(
                knot_vector_scaled, self.deg[i]))

            node_posi.append(
                knot_vector_scaled[self.deg[i]:len(knot_vector_scaled)-self.deg[i]])
            self.knot_vectors.append(knot_vector_scaled)

        self.vect_ctrl = ctrlpts_posi
        self.c = MeshGrid(ctrlpts_posi)
        self.nc = self.c.shape[0]

        self.n = MeshGrid(node_posi)

    def OperatorFFD(self, dim_node=1):
        if self.dim == 2:
            N = Get2dBasisFunctionsAtPts(self.npts[:, 0], self.npts[:, 1],
                                         self.knot_vectors[0], self.knot_vectors[1],
                                         self.deg[0], self.deg[1])
        elif self.dim == 3:
            N = Get3dBasisFunctionsAtPts(self.npts[:, 0], self.npts[:, 1], self.npts[:, 2],
                                         self.knot_vectors[0], self.knot_vectors[1], self.knot_vectors[2],
                                         self.deg[0], self.deg[1], self.deg[2])

        N_reduct = self.ClearControlPoint(N)
        # self.c_del = self.c[self.idc_del].copy()
        self.cr = self.c[self.idc]
        self.ncr = self.cr.shape[0]

        if self.precond:
            N_reduct = PreConditioning(N_reduct)
            self.Rs = N_reduct

        if dim_node != 1:
            N_reduct = sps.block_diag([N_reduct]*dim_node, format='csc')

            # idc = self.idc.copy()
            # for nb in range(1, dim_node):
            #     self.idc = np.concatenate((self.idc, idc+nb*self.nc))

        self.dim_node = dim_node
        print("FFD operator computed")
        return N_reduct

    def ClearControlPoint(self, N):
        sum_N = N.sum(axis=1)

        idc, _ = np.where(sum_N > self.limit)
        N_reduct = N[idc]

        self.idc = idc
        self.idc_del = np.delete(np.arange(len(sum_N)), idc)
        return N_reduct

    def PlotPoint(self, del_point=False):
        if self.dim == 2:
            plt.figure()
            plt.plot(self.npts[:, 0], self.npts[:, 1],
                     'k.', label='Input point cloud')
            plt.plot(self.c[:, 0], self.c[:,
                     1], 'bo', label='Control points')
            if del_point:
                plt.plot(self.c_del[:, 0], self.c_del[:,
                         1], 'ro', label='Deleted control points')
            plt.axis('equal')
            plt.legend()

    def Morphing(self, bc):
        self.c += bc
        self.npts += self.Rs.T@bc

    def SelectNodesBox(self):
        """
        Selection of all the nodes of a mesh lying in a box defined by two
        points clics.
        """
        plt.figure()
        self.PlotPoint()
        figManager = plt.get_current_fig_manager()
        if hasattr(figManager.window, 'showMaximized'):
            figManager.window.showMaximized()
        else:
            if hasattr(figManager.window, 'maximize'):
                figManager.resize(figManager.window.maximize())
        plt.title("Select 2 points... and press enter")
        pts1 = np.array(plt.ginput(2, timeout=0))
        plt.close()
        inside = (
            (self.c[:, 0] > pts1[0, 0])
            * (self.c[:, 0] < pts1[1, 0])
            * (self.c[:, 1] > pts1[1, 1])
            * (self.c[:, 1] < pts1[0, 1])
        )
        (nset,) = np.where(inside)
        self.PlotPoint()
        plt.plot(self.c[nset, 0], self.c[nset, 1], "ro")
        return nset

    def VTRwriter(self, file_name, data=dict()):
        x = self.vect_ctrl[0]
        y = self.vect_ctrl[1]
        if self.dim == 2:
            z = np.zeros(1)
        else:
            z = self.vect_ctrl[2]

        vtr = VTRWriter(x, y, z)

        if len(data) != 0:
            for key in data:
                vtr.addPointData(key, self.dim_node, data[key])
        vtr.VTRWriter(file_name)

    def Reshape(self, vect):
        vect[self.idc] = vect
        return vect
