"""
@author: Maxim Vavilin maxim.vavilin@kit.edu
"""

import os
import math
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import matplotlib as mpl
import treams
import copy
import time
import treams.special as sp

import repscat as rs

from repscat.tmatrix_diag import TmatrixDiagonal

import time

def plot_for_xi(radius, k_list, xi):
    jay_max = 3
    
    tmat = rs.get_silicon_tmat(radius, jay_max, k_list)

    k1_list = np.linspace(k_list[0]*np.exp(-2*np.abs(xi)), k_list[-1]*np.exp(2*np.abs(xi)), len(k_list))
    
    time_init = time.time() 
    tmat_boosted = tmat.boost_precise(xi, k1_list, len(k1_list))  # (k1,k2,lam1,lam2,j1,j2,m1,m2)
    # tmat_boosted = tmat.boost_tmat(xi, len(k1_list), len(k1_list))  # (k1,k2,lam1,lam2,j1,j2,m1,m2)
    print('Time: ', time.time() - time_init)
    
    lam1 = 1
    lam2 = 1
    jay1 = 1
    jay2 = 1
    m1 = 1
    m2 = 1
    
    # tmat_boosted.restore_diagonal(lam1,lam2,jay1,jay2,m1,m2)
    axis_limit = [k_list[0]*1000, k_list[-1]*1000]
    tmat_boosted.plot(axis_limit, axis_limit, lam1,lam2,jay1,jay2,m1,m2)
    plt.savefig(f'figs/tmat_element_{xi}.png', bbox_inches="tight")
    # plt.show()
    
def plot(radius, k_list):
    xis = [0.025, 0.05]
    for xi in xis:
        plot_for_xi(radius, k_list, xi)
    