"""
@author: Maxim Vavilin maxim@vavilin.de
"""
import numpy as np
import repscat as rs
import matplotlib.pyplot as plt


def get_k_diff_of_theta(ks_of_theta):
    k_diff_of_theta = np.zeros(ks_of_theta.shape)
    for i_theta in range(ks_of_theta.shape[0]):
        k_list = ks_of_theta[i_theta]
        k_diff_of_theta[i_theta]= np.diff(k_list, append=k_list[-1])
    return k_diff_of_theta



def plot_both_methods(incident_pw, xi, boosted_theta_list, boosted_ks_of_theta):
    boosted_pw = incident_pw.boost(xi)
    i_t = 1
    plt.plot(boosted_ks_of_theta[i_t], np.abs(incident_pw.vals[:, 0, i_t, 0]))
    plt.plot(boosted_pw.k_list, np.abs(boosted_pw.vals[:, 0, i_t, 0]),'--')
    
    # i_theta_ref = 100
    # k = 0.04
    # i_k_for_interpolated = 
    # k0_of_theta_list = np.zeros_like(boosted_theta_list)
    # for i_theta0 
    # idx_k_precise = np.abs(array - value).argmin()
    # idx_k_interpolated = np.abs(array - value).argmin()
    
    # plt.plot(boosted_theta_list, np.abs(incident_pw.vals[150, 0, :, 0]))
    # plt.plot(boosted_pw.theta_list, np.abs(boosted_pw.vals[150, 0, :, 0]),'--') # reference
    plt.show()

def compute_precise_energy_as_sums(ks_of_theta, thetas, phis, vals):

    phi_part = np.diff(phis, append=phis[-1])
    theta_part = np.diff(thetas, append=thetas[-1])*np.sin(thetas)
    res = 0 
    for i_theta, theta in np.ndenumerate(thetas):  
        k_list = ks_of_theta[i_theta]
        k_part = np.diff(k_list, append=k_list[-1])*k_list**2
        for i_lam in [0,1]:
            for i_phi, phi in np.ndenumerate(phis):
                for i_k, k in np.ndenumerate(k_list):
                    res += phi_part[i_phi]*theta_part[i_theta]*k_part[i_k]*np.abs(vals[i_k, i_lam, i_theta, i_phi])**2
    
    energy = res * rs.C_0_SI * rs.H_BAR_SI * 1e9
    
    return energy

def check_at_itheta(ks_of_theta, boosted_theta_list, vals, incident_pw, xi):
    phi_part = np.diff(incident_pw.phi_list, append=incident_pw.phi_list[-1])
    ######### Interpolated
    chunk_interp = 0
    boosted_pw = incident_pw.boost(xi)
    theta_part0 = np.diff(boosted_pw.theta_list, append=boosted_pw.theta_list[-1])*np.sin(boosted_pw.theta_list)
    k_list0 = boosted_pw.k_list
    k_part0 = np.diff(k_list0, append=k_list0[-1])*k_list0**2
    for i_lam in [0,1]:
        print(i_lam)
        for i_phi, phi in np.ndenumerate(incident_pw.phi_list):
            for i_theta, theta in np.ndenumerate(boosted_pw.theta_list):
                    for i_k, k in np.ndenumerate(k_list0):
                        chunk_interp += phi_part[i_phi]*theta_part0[i_theta]*k_part0[i_k]*np.abs(boosted_pw.vals[i_k, i_lam, i_theta, i_phi])**2
            
    ####### Precise
    chunk_precise = 0 
    
    theta_part1 = np.diff(boosted_theta_list, append=boosted_theta_list[-1])*np.sin(boosted_theta_list)
    
    for i_theta, theta in np.ndenumerate(boosted_theta_list):
        k_list = ks_of_theta[i_theta]
        k_part = np.diff(k_list, append=k_list[-1])*k_list**2
        for i_lam in [0,1]:
            for i_phi, phi in np.ndenumerate(incident_pw.phi_list):
                for i_k, k in np.ndenumerate(k_list):
                    chunk_precise += phi_part[i_phi]*theta_part1[i_theta]*k_part[i_k]*np.abs(vals[i_k, i_lam, i_theta, i_phi])**2
    
    print("chunk_precise", chunk_precise * rs.C_0_SI * rs.H_BAR_SI * 1e9)
    print("chunk_interp", chunk_interp * rs.C_0_SI * rs.H_BAR_SI * 1e9)
    

def compute_precise_energy(ks_of_theta, thetas, phis, vals):
    k_diff_of_theta = get_k_diff_of_theta(ks_of_theta)

    energy = (
        np.einsum(
            "kltp,tk,tk,t,p->",
            np.square(np.abs(vals)),
            np.square(ks_of_theta),
            k_diff_of_theta,
            np.diff(thetas, append=thetas[-1]) * np.sin(thetas),
            np.diff(phis, append=phis[-1]),
            optimize=True
        )
        * rs.C_0_SI
        * rs.H_BAR_SI
        * 1e9
    )
    return energy

def quantities(incident_pw, xi):

    boosted_ks_of_theta, boosted_theta_list = incident_pw._boost_domain(xi)
    
    # energy = compute_precise_energy(boosted_k_array, boosted_theta_list, incident_pw.phi_list, incident_pw.vals)
    # energy_sums = compute_precise_energy_as_sums(boosted_k_array, boosted_theta_list, incident_pw.phi_list, incident_pw.vals)
    
    plot_both_methods(incident_pw, xi, boosted_theta_list, boosted_ks_of_theta)
    
    
    # print('Precize energy: ', energy)
    # print('Precize energy from sum: ', energy_sums) 
    
    
    # i_lam = 0
    # i_phi = 50
    # i_theta = 3
    # check_at_itheta(boosted_k_array, boosted_theta_list, incident_pw.vals, incident_pw, xi)
    
    