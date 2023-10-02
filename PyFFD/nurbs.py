# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 10:55:03 2023

@author: SENTAGNE
"""
import numpy as np
from numba import njit
import scipy.sparse as sps

# %% Commun tools


@njit
def findKnotSpan(u, U, p):
    """
    Finds the knots space of a given knot parameter u in
    the knot vector U corresponding to the degree p 
    """
    m = len(U)
    # if u==U[m-p-1]:
    if np.abs(u-U[m-p-1]) < 1e-14:
        k = m-p-2
    else:
        for k in range(len(U)):
            if u < U[k]:
                if U[k] - u != 0:
                    k -= 1
                break
    return k


@njit
def derbasisfuns(i, pl, U, nders, u):
    """ 
    i : knot span of u 
    p : degree 
    u : parameter on which we want to evaluate the function 
    nders: number of derivatives 
    U : knot vector 

    # i span de u 
    # pl = degrés de la nurbs
    # u = endroit ou l'on veut la fonction
    # nders = numéro de la dérivée désirée
    # U = vecteur de noeud de la fonction """

#    import pdb; pdb.set_trace()
    u_knotl = U.copy()
    left = np.zeros((pl+1))
    right = np.zeros((pl+1))
    ndu = np.zeros((pl+1, pl+1))
    ders = np.zeros((nders+1, pl+1))
    ndu[0, 0] = 1
    for j in range(pl):  # 1:pl
        left[j+1] = u - u_knotl[i-j]  # rq Ali : i-j au lieu de i-j-1
        right[j+1] = u_knotl[i+j+1] - u  # rq : i+j+1 au lieu de i+j
        saved = 0
        for r in range(j+1):  # 0:j-1
            ndu[j+1, r] = right[r+1] + left[j-r+1]
            temp = ndu[r, j]/ndu[j+1, r]
            ndu[r, j+1] = saved + right[r+1]*temp
            saved = left[j-r+1]*temp
        ndu[j+1, j+1] = saved
#    print('checkpoint1 : '+str(ndu))

        # load basis functions
    for j in range(pl+1):  # 0:pl
        ders[0, j] = ndu[j, pl]
#    print('checkpoint2 : '+str(ders))

        # compute derivatives
    for r in range(pl+1):  # 0:pl              # loop over function index
        s1 = 0
        s2 = 1                # alternate rows in array a
        a = np.zeros((nders+1, nders+1))
        a[0, 0] = 1
        # loop to compute kth derivative
        for k in range(nders):   # 1:nders
            d = 0
            rk = r-(k+1)
            pk = pl-(k+1)
            if (r >= (k+1)):
                a[s2, 0] = a[s1, 0]/ndu[pk+1, rk]
                d = a[s2, 0]*ndu[rk, pk]
            if (rk >= -1):
                j1 = 1
            else:
                j1 = -rk
            if ((r-1) <= pk):
                j2 = k
            else:
                j2 = pl-r
            for j in np.arange(j1, j2+0.1):  # j1:j2
                j = int(j)
                a[s2, j] = (a[s1, j] - a[s1, j-1])/ndu[pk+1, rk+j]
                d = d + a[s2, j]*ndu[rk+j, pk]
            if (r <= pk):
                a[s2, k+1] = -a[s1, k]/ndu[pk+1, r]
                d = d + a[s2, k+1]*ndu[r, pk]
            ders[k+1, r] = d
            j = s1
            s1 = s2
            s2 = j            # switch rows

    #     Multiply through by the correct factors

    r = pl
    for k in range(nders):   # 1:nders
        for j in range(pl+1):   # 0:pl
            ders[k+1, j] = ders[k+1, j]*r
        r = r*(pl-(k+1))

    return ders


@njit
def BasisFunc(i, u, p, U):
    """
    Evaluates the non zero basis functions N_{i,p}(u)
    for u living in the knot span [U_i,U_{i+1}[
    From the Nurbs Book ( Les Piegl, Wayne Tiller)
    """
    N = np.zeros(p+1)
    left = np.zeros(p+1)
    right = np.zeros(p+1)
    N[0] = 1.
    for j in range(1, p+1):
        # print(i+j)
        left[j] = u-U[i+1-j]
        right[j] = U[i+j]-u
        saved = 0.
        for r in range(j):
            temp = N[r]/(right[r+1]+left[j-r])
            N[r] = saved+right[r+1]*temp
            saved = left[j-r]*temp
        N[j] = saved
    return N

# %% 2D B-spline box tools


def Get2dBasisFunctionsAtPts(u, v, U, V, p, q):
    nb_u_values = len(u)
    nnz_values = nb_u_values*(p+1)*(q+1)
    indexI = np.zeros(nnz_values)
    indexJ = np.zeros(nnz_values)
    valuesN = np.zeros(nnz_values)

    n = len(U) - 1 - p
    m = len(V) - 1 - q
    nbf = n*m

    indexI, indexJ, valuesN = KronLoop(
        u, v, U, V, p, q, nb_u_values, n, indexI, indexJ, valuesN)

    phi = sps.csc_matrix((valuesN, (indexI, indexJ)), shape=(nb_u_values, nbf))
    return phi.T


@njit
def KronLoop(u, v, U, V, p, q, nb_u_values, n, indexI, indexJ, valuesN):
    l = 0
    for k in range(nb_u_values):
        spanu = findKnotSpan(u[k], U, p)
        spanv = findKnotSpan(v[k], V, q)
        Nu = derbasisfuns(spanu, p, U, 1, u[k])
        Nv = derbasisfuns(spanv, q, V, 1, v[k])
        for j in range(q+1):
            for i in range(p+1):
                valuesN[l] = Nv[0][j]*Nu[0][i]
                indexI[l] = k
                indexJ[l] = spanu - p + i + (spanv - q + j)*n
                l = l+1
    return indexI, indexJ, valuesN


# %% 3D B-spline box tools
def Get3dBasisFunctionsAtPts(x, y, z, Xi, Eta, Zeta, p, q, r):
    nb_x_values = len(x)
    # Number of basis functions that support a point
    nbf_pt = (p+1)*(q+1)*(r+1)
    nnz_values = nb_x_values*nbf_pt
    indexI = np.zeros(nnz_values)
    indexJ = np.zeros(nnz_values)
    valuesN = np.zeros(nnz_values)

    nbf_xi = len(Xi) - 1 - p
    nbf_eta = len(Eta) - 1 - q
    nbf_zeta = len(Zeta) - 1 - r

    nbf = nbf_xi * nbf_eta * nbf_zeta

    index = np.arange(nbf_pt)
    index_i = np.kron(np.ones((r+1)*(q+1)), np.arange(p+1))
    index_j = np.kron(np.ones(r+1), np.kron(np.arange(q+1), np.ones(p+1)))
    index_k = np.kron(np.arange(r+1), np.ones((p+1)*(q+1)))

    # ll=0  # uncomment if unvectorized version is used
    for up in range(nb_x_values):
        # Loop over the unstructured points
        spanx = findKnotSpan(x[up], Xi, p)
        spany = findKnotSpan(y[up], Eta, q)
        spanz = findKnotSpan(z[up], Zeta, r)

        # print(up, x[up], spanx, len(Xi))

        Nxi = BasisFunc(spanx, x[up], p, Xi)
        Neta = BasisFunc(spany, y[up], q, Eta)
        Nzeta = BasisFunc(spanz, z[up], r, Zeta)
        valuesN[index + up*nbf_pt] = np.kron(Nzeta, np.kron(Neta, Nxi))
        indexI[index + up*nbf_pt] = up
        indexJ[index + up*nbf_pt] = (spanx-p+index_i) + (spany -
                                                         q+index_j)*nbf_xi + (spanz-r+index_k)*nbf_xi*nbf_eta
        # Structured grid: i+j*nx+k*nx*ny (classic arrangement)
        # Non vectorized approach
        # for k in range(r+1):
        #     for j in range(q+1):
        #         for i in range(p+1):
        #             valuesN[ll]    = Nw[k]*Nv[j]*Nu[i]
        #             indexI[ll] = up
        #             # Structured grid: i+j*nx+k*nx*ny
        #             indexJ[ll] = spanu-p+i +(spanv-q+j)*n +(spanw-r+k)*n*m
        #             ll = ll +1
    phi = sps.csc_matrix((valuesN, (indexI, indexJ)),
                         shape=(nb_x_values, nbf))
    return phi.T
