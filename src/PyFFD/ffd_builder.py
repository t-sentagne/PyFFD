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
    def __init__(self, npts, size, deg, morp=False):
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
        self.morp = morp
        self.dim = npts.shape[1]
        if type(deg) != list and type(deg) != np.ndarray:  # FIXME: test pas assez générales
            deg = [deg,] * self.dim
        self.deg = deg
        if type(size) != list and type(size) != np.ndarray:
            size = [size,] * self.dim
        self.size = size

        self.e = []
        self.c = []

        self.crtpts_del = []
        self.limit = 1e-10
        self.precond = True
        self.SetMarginThickness()
        
        self.cr = []
        self.ncr = 0

    def SetThreshold(self, limit):
        self.limit = limit

    def SetPreconditioning(self, precond):
        if type(precond) != bool:
            raise NameError("Boolean is expected for 'precond' parameter")
        self.precond = precond

    def SetMarginThickness(self, delta=0):
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
            coord_min = self.npts[:, i].min() - self.delta[i]
            coord_max = self.npts[:, i].max() + self.delta[i]
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
        self.idc = np.arange(len(self.c))
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
        
        
        if self.morp:
            N_reduct = N
        else:
            N_reduct = self.ClearControlPoint(N)
            self.cr = self.c[self.idc]
            self.ncr = self.cr.shape[0]

        if self.precond and self.morp != True:
            print("precond")
            N_reduct = PreConditioning(N_reduct)

        if dim_node != 1:
            N_reduct = sps.block_diag([N_reduct]*dim_node, format='csc')

            # idc = self.idc.copy()
            # for nb in range(1, dim_node):
            #     self.idc = np.concatenate((self.idc, idc+nb*self.nc))

        self.dim_node = dim_node
        self.Rs = N_reduct
        print("FFD operator computed")
        return N_reduct

    def ClearControlPoint(self, N):
        sum_N = N.sum(axis=1)

        idc, _ = np.where(sum_N > self.limit)
        N_reduct = N[idc]

        self.idc = idc
        return N_reduct

    def GetDeletedControlPoints(self):
        return np.delete(np.arange(len(self.c)), self.idc)

    def PlotPoint(self, U=None, s=1, del_point=False):
        if U is None:
            U = np.zeros((len(self.idc), self.dim))
        xc = self.c[self.idc] + s * U
        disp = self.Rs.T @ U
        xn = self.npts + s * disp
        if self.dim == 2:
            plt.figure()
            plt.plot(xn[:, 0], xn[:, 1],
                     'k.', label='Input point cloud')
            plt.plot(xc[:, 0], xc[:, 1], 'bo', label='Control points')
            if del_point:
                idc_del = self.GetDeletedControlPoints()
                plt.plot(self.c[idc_del, 0], self.c[idc_del,
                         1], 'ro', label='Deleted control points')
            plt.axis('equal')
            plt.legend()
        else:
            ax = plt.figure().add_subplot(projection='3d')
            plt.plot(xn[:, 0], xn[:, 1], xn[:, 2],
                     'k.', label='Input point cloud')
            plt.plot(xc[:, 0], xc[:, 1], xc[:, 2],
                     'bo', label='Control points')
            if del_point:
                idc_del = self.GetDeletedControlPoints()
                plt.plot(self.c[idc_del, 0], self.c[idc_del, 1],
                         self.c[idc_del, 1],
                         'ro', label='Deleted control points')
            X = xc[:, 0]
            Y = xc[:, 1]
            Z = xc[:, 2]
            max_range = np.array(
                [X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()/2.0
            mid_x = (X.max()+X.min()) * 0.5
            mid_y = (Y.max()+Y.min()) * 0.5
            mid_z = (Z.max()+Z.min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            
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

        data_del = np.zeros(self.nc)
        idc_del = self.GetDeletedControlPoints()
        data_del[idc_del] = 1
        vtr.addPointData("Deleted points", 1, data_del)
        vtr.VTRWriter(file_name)

    def Reshape(self, vect):
        vect[self.idc] = vect
        return vect
