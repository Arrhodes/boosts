"""
@author: Maxim Vavilin maxim@vavilin.de
"""
import numpy as np
import repscat as rs
import matplotlib.pyplot as plt
import matplotlib as mpl

def transfer_no_boost(incident, radius):
    ## Transfer without boost
    jay_max = incident.jay_max
    k2_list = incident.k_list
    
    tmat_rest = rs.get_silicon_tmat(radius, jay_max, k2_list) # input k2
    # tmat.plot()
    # plt.show()

    # print('Energy in pulse = ', incident.energy())
    quantities = tmat_rest.compute_transfer(incident)
    print('Rest: Transfer energy = ', np.real(quantities["energy"]))
    print('Rest: Transfer momentum = ', np.real(quantities["momentum_z"]))


def transfer_boosted_in_lab_frame(incident_am, radius, xi):
    # Here we boost the T-matrix
    jay_max = incident_am.jay_max
    k2_list = incident_am.k_list
    k_list = rs.boost_wavenumbers(k2_list, xi)
    tmat = rs.get_silicon_tmat(radius, jay_max, k_list)
    tmat_boosted = tmat.boost_tmat_precise(xi)  # for input k2_list

    # lam1 = 1
    # lam2 = 1
    # jay1 = 1
    # jay2 = 1
    # m1 = 1
    # m2 = 1
    # tmat_boosted.plot(lam1,lam2,jay1,jay2,m1,m2)
    # plt.show()

    # k2_lists = tmat_boosted.k2_lists
    # jay_max = tmat_boosted.jay_max
    # i_k1 = 0
    # plt.plot(k2_lists[i_k1], tmat_boosted.vals[i_k1,:,0,0,0,0,jay_max,jay_max])
    # plt.show()

    ## Compute transfer over boosted Tmat
    # incident_stretch = incident_am.interpolate(tmat_boosted._k2_list)

    quantities = tmat_boosted.compute_transfer(incident_am)


    print(f'Boosted, lab frame: E transfer = {np.real(quantities["energy"])}')
    print(f'Boosted, lab frame: P transfer = {np.real(quantities["momentum_z"])}')

    return quantities

def transfer_boosted_object_frame(incident_pw, jay_max, radius, xi):
    # Here we boost fields backwads
    incident_pw_boosted = incident_pw.boost(-xi) # Minus here
    # incident_pw_boosted.plot()
    # incident_pw.plot()
    # plt.show()

    # incident



    incident_pw_boosted_am = incident_pw_boosted.in_angular_momentum_basis(jay_max)
    # incident_pw_boosted_am.plot()
    # plt.show()

    k2_list = incident_pw_boosted_am.k_list

    tmat_rest = rs.get_silicon_tmat(radius, jay_max, k2_list)
    quantities = tmat_rest.compute_transfer(incident_pw_boosted_am)

    print(f'Boosted, object frame: E transfer = {np.real(quantities["energy"])}')
    print(f'Boosted, object frame: P transfer = {np.real(quantities["momentum_z"])}')

    return quantities


