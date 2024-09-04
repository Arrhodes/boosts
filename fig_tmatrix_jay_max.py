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

def get_wavenumbers_needed_for_boosts(k_list, xi_max):
    """Computes the maximal domain spreading of the initial pulse after boost"""

    k_full_min = k_list[0] * np.exp(-np.abs(xi_max))
    k_full_max = k_list[-1] * np.exp(np.abs(xi_max))

    return np.linspace(k_full_min, k_full_max, len(k_list))

def plot(tmat_name, jay_max, k_list_unboosted_pulse, xi_max):
    plt.rcParams.update(
        {
            "font.size": 18,
        }
    )
    wavenumbers_needed_for_boosts = get_wavenumbers_needed_for_boosts(k_list_unboosted_pulse, xi_max)
    
    fig, ax = plt.subplots()
    
    for jay in range(1, jay_max + 2):
        if tmat_name == 'silicon':
            radius = 150
            tmat = rs.get_silicon_tmat(radius, jay, wavenumbers_needed_for_boosts)
        elif tmat_name == 'gold_cluster':
            radius = 50
            distance = 100
            tmat = rs.get_gold_cluster_tmat(radius, distance, jay, wavenumbers_needed_for_boosts)
        elif tmat_name == 'gold_sphere':
            radius = 150
            tmat = rs.get_gold_sphere_tmat(radius, jay, wavenumbers_needed_for_boosts)
        max_str = "\\text{max}"
        ax = tmat.plot_norm(ax, label=f"$j_{max_str}=$ {jay}", linestyle="-")

    ax.set_xlabel("Wavenumber $k$, $\\text{$\mu$m}^{-1}$")
    ax.set_ylabel("$||T(k)||$")

    legend = plt.legend(frameon=1, prop={'size': 15})
    frame = legend.get_frame()
    frame.set_color("white")
    frame.set_alpha(1)
    
    plt.grid(True)

    # plt.show()
    plt.savefig(f"figs/tmatrix_jay_max.png", bbox_inches='tight')
    
