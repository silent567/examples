#!/usr/bin/env python
# coding=utf-8

import numpy as np
from pygfl.easy import solve_gfl

def sparsemax(z,gamma=1):
    z = z / gamma
    z_sorted = np.sort(z)[::-1]
    cumsum = np.cumsum(z_sorted)
    k = 0
    while k<z.size and 1+(k+1)*z_sorted[k] > cumsum[k]:
        k += 1
    tau = (cumsum[k-1]-1)/(k)
    return np.maximum(z-tau,0)

def softmax(z):
    z_exp = np.exp(z)
    return z_exp/np.sum(z_exp)

def gfusedlasso(z,A,lam=None):
    A = np.triu(A) > 0
    edges = np.stack(np.mask_indices(A.shape[0],lambda n,k:A),axis=-1)
    z_fused = solve_gfl(z,edges,lam=lam)
    return z_fused

def gfusedmax(z,A,lam=None,gamma=1):
    z_fused = gfusedlasso(z,A,lam)
    return sparsemax(z_fused,gamma)


