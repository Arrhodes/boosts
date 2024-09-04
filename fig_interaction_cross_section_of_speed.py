"""
@author: Maxim Vavilin maxim.vavilin@kit.edu
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


def plot(tmat_name, central_wavelength, xi_max, len_xi, radius, jay_max, ax=None):
    '''Plot for article: Interaction cross-section as function of v/c'''
    
    if ax is None:
        fig, ax = plt.subplots()
        
    k0 = 2*np.pi/central_wavelength
    rapidities = np.linspace(-xi_max, xi_max, len_xi)
    kays = k0*np.exp(-rapidities) # pulse seen inversely boosted
    
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
    
    norms =  np.einsum(
            "kabijmn,kabijmn,k->k",
            np.conj(tmat.vals),  # (k,a,b,i,j,m,n)
            tmat.vals,  # (k,a,b,i,j,m,n)
            (np.pi)/tmat.k_list**2, # using Karim and my convention
            optimize=True
        )
    
    linecolor = 'grey'
    
    velocities = np.tanh(rapidities)
    ax.plot(velocities, np.real(norms)*1e-18, linestyle='-', color=linecolor, zorder=2.5)

    # ax.set_ylabel("Interaction cross-section $||T(k)||^2$", color=linecolor)
    ax.set_ylabel(r"$\sigma^\text{aver}_\text{sca}$, $\text{m}^2$", color=linecolor)
    ax.tick_params(axis='y', labelcolor=linecolor)

    # plt.show()
    # plt.savefig(f"InteractionCS_total.png", bbox_inches='tight')
    
    return ax

