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


from fig_transfer_both_frames import get_transfer_comoving, boost, plot_momentum_lab

def print_res_for(xi, incident, jay_max, radius):
    
    k2_list = incident.k_list 
    k_list = np.linspace(k2_list[0]*np.exp(-np.abs(xi)), k2_list[-1]*np.exp(np.abs(xi)), len(k2_list)) 
    k1_list = np.linspace(k2_list[0]*np.exp(-2*np.abs(xi)), k2_list[-1]*np.exp(2*np.abs(xi)), len(k2_list))
    
    tmat = rs.get_silicon_tmat(radius, jay_max, k_list)

    t_init = time.time()
    tmat_boosted = tmat.boost_precise(xi, k1_list, len(k_list))  # (k1,k2,lam1,lam2,j1,j2,m1,m2)
    print('Time tmat boost:', time.time() - t_init)
    
    _, momentum = tmat_boosted.compute_transfer(incident)
    
    print(f"P transfer = {momentum}")
    return momentum

    
def get_tmat_momentum_diffs(incident, jay_max, radius, xi_list):
    tmat_diffs = []
    for xi in xi_list:
        tmat_diffs.append(print_res_for(xi, incident, jay_max, radius))
    
    print('tmat_diffs')
    print(tmat_diffs)
    return tmat_diffs

def fig_scatter_from_tmat(incident_pw, jay_max, radius, xi_list, ax = None):
    if ax is None:
        fig, ax = plt.subplots()
        
    tmat_momentum_diffs = get_tmat_momentum_diffs(incident_pw, jay_max, radius, xi_list)
    ax.scatter(np.tanh(xi_list), tmat_momentum_diffs)
    
    return ax


def plot(incident_pw, jay_max, radius):
    ''' Incident wavenumbers in lab frame: k2_list,  
        Incident and scattered wavenumbers in obj frame: k_list,
        Scattered wavenumbers in lab frame: k1_list '''
        
    fig, ax = plt.subplots()
    
    xi_list_tmat = [0.00001, 0.025, 0.05, 0.075, 0.1, 0.125]
    # xi_list_tmat = [0.125]
    fig_scatter_from_tmat(incident_pw, jay_max, radius, xi_list_tmat, ax)
    
    xi_list_frame_hop = np.linspace(-0.005, 0.155, 100)
    energy_diff_comoving, momentum_diff_comoving = get_transfer_comoving('silicon', incident_pw, jay_max, xi_list_frame_hop)
    _, momentum_diff_lab = boost(energy_diff_comoving, momentum_diff_comoving, xi_list_frame_hop)
    plot_momentum_lab('via_boosted_tmat', momentum_diff_lab, xi_list_frame_hop, ax)
    
    fig.savefig(f'figs/Tmat_lab_transfer_r={radius}_j={jay_max}.png')
    # plt.show()
    
    
    

    
    
    
    
    