"""
@author: Maxim Vavilin maxim@vavilin.de
"""
import numpy as np
import matplotlib.pyplot as plt

import repscat as rs

def boost(energy, momentum, xi):
        energy_new = (
            np.cosh(xi) * energy + np.sinh(xi) * momentum * rs.C_0_SI
        )
        mom_new = (
            np.sinh(xi) * energy / rs.C_0_SI + np.cosh(xi) * momentum
        )
        
        return energy_new, mom_new

def compute_transfer_from_boosting_fields(incident_pw, jay_max, radius, xi):
    incident_pw = incident_pw.boost(xi)
    # incident_am = incident_pw.in_angular_momentum_basis(jay_max)

    # tmat = rs.get_silicon_tmat(radius, jay_max, incident_am.k_list)

    # scattered_am = tmat.scatter(incident_am)

    # scattered_pw = scattered_am.in_plane_wave_basis(
    #     incident_pw.theta_list, incident_pw.phi_list
    # )
    
    # outgoing_pw = scattered_pw + incident_pw
    
    # incident_pw_lab = incident_pw.boost(xi)
    # outgoing_pw_lab = outgoing_pw.boost(xi)
    
    print("Method: using plane waves")
    print("Energy = ", incident_pw.energy())
    print("Momentum = ", incident_pw.momentum_z())
    
    # print('Momentum Pout - Pin = ', outgoing_pw_lab.momentum_z() - incident_pw_lab.momentum_z())
    # print('Energy Eout - Ein = ', outgoing_pw_lab.energy() - incident_pw_lab.energy())

def compute_transfer_from_boosting_quantities(incident_pw, jay_max, radius, xi):
    # incident_pw = incident_pw.boost(-xi)
    # incident_am = incident_pw.in_angular_momentum_basis(jay_max)

    # tmat = rs.get_silicon_tmat(radius, jay_max, incident_am.k_list)

    # scattered_am = tmat.scatter(incident_am)

    # scattered_pw = scattered_am.in_plane_wave_basis(
    #     incident_pw.theta_list, incident_pw.phi_list
    # )
    
    # outgoing_pw = scattered_pw + incident_pw
    
    # incident_pw_lab = incident_pw.boost(xi)
    # outgoing_pw_lab = outgoing_pw.boost(xi)
    
    print("Method: boost quantities")
    energy = incident_pw.energy()
    momentum = incident_pw.momentum_z()
    
    energy_obj, momentum_obj = boost(energy, momentum, xi)
    
    print("Energy = ", energy_obj)
    print("Momentum = ", momentum_obj)

def print_result(incident_pw, jay_max, radius, xi):
    with plt.style.context('ggplot'):
        compute_transfer_from_boosting_fields(incident_pw, jay_max, radius, xi)
        compute_transfer_from_boosting_quantities(incident_pw, jay_max, radius, xi)
        