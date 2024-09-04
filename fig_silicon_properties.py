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

def plot_wavenumber_content(ax, incident, xi_max):
    # As perceived in co-moving frame => invert xi
    if ax is None:
        fig, ax = plt.subplots()
        
    field_left = incident.boost(-xi_max)
    field_right = incident.boost(xi_max)
    
    color = 'darkred'
    
    ax.plot(field_left.k_list*1000, field_left.get_wavenumber_photon_density(), color = color)
    ax.plot(incident.k_list*1000, incident.get_wavenumber_photon_density(), color = color)
    ax.plot(field_right.k_list*1000, field_right.get_wavenumber_photon_density(), color = color)
    
    ax.set_ylabel("Photons per wavenumber, $\\text{$\mu$m}$", color=color)
    ax.tick_params(axis='y', labelcolor=color)
    

    textstr_left = "$v=0.8c$"
    textstr_center = "$v=0$"
    textstr_right = "$v=-0.8c$"
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)  # Textbox style
    ax.text(0.15, 0.83, textstr_left, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='center')
    ax.text(0.33, 0.4, textstr_center, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='center')
    ax.text(0.75, 0.14, textstr_right, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='center')
    
    return ax

def get_wavenumbers_needed_for_boosts(k_list, xi_max):
    """Computes the maximal domain spreading of the initial pulse after boost"""

    k_full_min = k_list[0] * np.exp(-np.abs(xi_max))
    k_full_max = k_list[-1] * np.exp(np.abs(xi_max))

    return np.linspace(k_full_min, k_full_max, len(k_list))

def plot(k_list_unboosted_pulse, xi_max, incident_pw):
    wavenumbers_needed_for_boosts = get_wavenumbers_needed_for_boosts(k_list_unboosted_pulse, xi_max)
    plt.rcParams.update(
        {
            "font.size": 18,
        }
    )
    
    fig, ax = plt.subplots()
    
    silicon = rs.Silicon()
    silicon.plot(ax, wavenumbers_needed_for_boosts)

    plot_wavenumber_content(ax.twinx(), incident_pw, xi_max)
    
    ax.set_xlabel("Wavenumber $k$, $\\text{$\mu$m}^{-1}$")
    ax.set_ylabel("Silicon properties")

    # legend = ax.legend(frameon=1)
    legend = ax.legend(frameon=1, loc='upper right') #, bbox_to_anchor=(0.05, 1)
    frame = legend.get_frame()
    frame.set_color("white")
    frame.set_alpha(0)
    
    # plt.grid(True)

    # plt.show()
    plt.savefig(f"figs/Silicon_wide.png", bbox_inches='tight')
