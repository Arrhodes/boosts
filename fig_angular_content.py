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


from matplotlib.ticker import FormatStrFormatter, MultipleLocator

# plt.style.use("seaborn-v0_8-whitegrid")

def plot_difference_of_angular_photon_density(rep_one_pw, rep_two_pw):

    fig, ax = plt.subplots()    
    k_list = rep_one_pw._k_list
    theta_list = rep_one_pw._theta_list
    phi_list = rep_one_pw._phi_list

    # Legend: k wavenumber, l helicity, t theta, p phi
    angular_photon_density_one = np.einsum(
        "kltp,k,t,p->t",
        np.square(np.abs(rep_one_pw.vals)),
        np.diff(k_list, append=k_list[-1]) * k_list,
        np.sin(theta_list),
        np.diff(phi_list, append=phi_list[-1]),
    )
    
    angular_photon_density_two = np.einsum(
        "kltp,k,t,p->t",
        np.square(np.abs(rep_two_pw.vals)),
        np.diff(k_list, append=k_list[-1]) * k_list,
        np.sin(theta_list),
        np.diff(phi_list, append=phi_list[-1]),
    )
    
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g $\pi$'))
    ax.xaxis.set_major_locator(MultipleLocator(base=0.25))
    labels = [r'$-\pi/4$', '$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
    ax.set_xticklabels(labels)
    
    ax.set_ylabel(r"Angular photon density $N_\theta$")
    ax.set_xlabel(r"Polar angle $\theta$")
    
    plt.plot(rep_one_pw._theta_list/np.pi, angular_photon_density_two-angular_photon_density_one)
    
    plt.show()

def plot(incident_pw, jay_max, radius, xi):
    '''Plot for article: Angular content in objects frame'''
    # incident_pw = incident_pw.boost(-xi)
    incident_am = incident_pw.in_angular_momentum_basis(jay_max)

    tmat = rs.get_silicon_tmat(radius, jay_max, incident_am.k_list)

    scattered_am = tmat.scatter(incident_am)

    scattered_pw = scattered_am.in_plane_wave_basis(
        incident_pw.theta_list, incident_pw.phi_list
    )
    
    # scattered_pw.plot_angular_photon_density()
    
    outgoing_pw = scattered_pw + incident_pw
    
    incident_pw_lab = incident_pw.boost(xi)
    outgoing_pw_lab = outgoing_pw.boost(xi)
    
    print('Momentum Pout - Pin = ', outgoing_pw.momentum_z() - incident_pw.momentum_z())
    print('Energy Eout - Ein = ', outgoing_pw.energy() - incident_pw.energy())
    
    # print('Momentum Pout - Pin = ', outgoing_pw_lab.momentum_z() - incident_pw_lab.momentum_z())
    # print('Energy Eout - Ein = ', outgoing_pw_lab.energy() - incident_pw_lab.energy())
    
    plot_difference_of_angular_photon_density(incident_pw, outgoing_pw)
    # plot_difference_of_angular_photon_density(incident_pw_lab, outgoing_pw_lab)
    
    
    # outgoing_pw.plot_angular_photon_density()
    # incident_pw.plot_angular_photon_density()
    
    
    
