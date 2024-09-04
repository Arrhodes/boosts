"""
@author: Maxim Vavilin maxim@vavilin.de
"""

import os
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import matplotlib as mpl
import treams
import copy
import time
import treams.special as sp

import repscat as rs




def get_identity(shape):

    identity = np.zeros(shape, dtype=complex)
    
    for idx0 in range(shape[0]):
        for idx1 in range(shape[2]):
            for idx2 in range(shape[4]):
                for idx3 in range(shape[6]):
                    identity[idx0, idx1, idx1, idx2, idx2, idx3, idx3] = 1
    
    return identity

def get_absorption(tmat):
    identity = get_identity(tmat.vals.shape)
    
    # one_minus_t = identity - tmat.vals/2
    
    # extinction = np.einsum(
    #         "kaaiimm,k->k",
    #         np.real(tmat.vals)/2,
    #         (4*np.pi)/tmat.k_list**2,
    #         optimize=True
    #     )
    
    # trace = np.einsum(
    #         "kaaiimm,kaaiimm,k->k",
    #         tmat.vals/2,
    #         np.conj(one_minus_t),  # (k,a,b,i,j,m,n)
    #         (4*np.pi)/tmat.k_list**2,
    #         optimize=True
    #     )
    
    smat_vals = identity + tmat.vals
    
    s_dagger_s = np.einsum(
        'kxaycze,kxbydzf->kabcdef',
        np.conj(smat_vals), # transposed
        smat_vals, # (k,a,b,i,j,m,n)
        optimize = True
    )
    
    absorption_mat = identity - s_dagger_s
    

    trace = np.einsum(
            "kaaiimm,k->k",
            absorption_mat,  # (k,a,b,i,j,m,n)
            (4*np.pi)/tmat.k_list**2,
            optimize=True
        )
    
    # j_max = tmat_vals.shape[3]
    # normalization = (j_max+1)**2 - 1
    
    return trace


def plot(tmat_name, central_wavelength, xi_max, len_xi, radius, jay_max, ax=None):
    """Plot for article: Interaction cross-section as function of v/c"""

    if ax is None:
        fig, ax = plt.subplots()

    k0 = 2 * np.pi / central_wavelength
    rapidities = np.linspace(-xi_max, xi_max, len_xi)
    kays = k0 * np.exp(-rapidities)  # pulse seen inversely boosted
    
    if tmat_name == 'silicon':
        radius = 150
        tmat = rs.get_silicon_tmat(radius, jay_max, kays)
    elif tmat_name == 'gold_cluster':
        radius = 50
        distance = 100
        tmat = rs.get_gold_cluster_tmat(radius, distance, jay_max, kays)
    elif tmat_name == 'gold_sphere':
        radius = 150
        tmat = rs.get_gold_sphere_tmat(radius, jay_max, kays)
    
    absorption = get_absorption(tmat)

    linecolor = 'grey'

    velocities = np.tanh(rapidities)
    ax.plot(velocities, np.real(absorption)*1e-18, linestyle='-', color=linecolor, zorder=2.5)

    # ax.set_ylabel("Absorption", color=linecolor)
    ax.set_ylabel(r"$\sigma^\text{aver}_\text{abs}$, $\text{m}^2$", color=linecolor)
    ax.tick_params(axis='y', labelcolor=linecolor)

    # plt.show()
    # plt.savefig(f"figs/Absorption_total.png", bbox_inches='tight')

    return ax
