"""
@author: Maxim Vavilin maxim@vavilin.de
"""
import numpy as np
import repscat as rs
from repscat.wavefunction_angular_momentum import WaveFunctionAngularMomentum
import matplotlib.pyplot as plt

def get_k_list_for_pulse(num_k, width_time, center_wavelength):
    # Will be referenced as incident k2_list in boosting context
    k0_center = 2*np.pi/center_wavelength
    width_k = 1/(rs.C_0*width_time)
    k2_min = k0_center-1.8*np.sqrt(2)*width_k # 2.7 before
    k2_max = k0_center+1.8*np.sqrt(2)*width_k
    return np.linspace(k2_min, k2_max, num_k)

def get_theta_list_new(num_theta, theta_fac):
    theta_threshold = 0.001
    theta_almost_max = np.sqrt(-2 * theta_fac**2 * np.log(theta_threshold))
    if theta_almost_max > np.pi:
        print ("Warning: theta_max > pi")

    theta_max = min(theta_almost_max, np.pi)
    
    theta_min = 0
    
    return np.linspace(theta_min, theta_max, num_theta)

def get_pulse(num_k, num_phi, num_theta, norm_fac, width_time, width_theta, central_wavelength):
    ### In PW basis
    k_list = get_k_list_for_pulse(num_k, width_time, central_wavelength)
    theta_list = get_theta_list_new(num_theta, width_theta)
    phi_list = np.linspace(0, 2 * np.pi, num_phi)
    
    k0 = 2 * np.pi / central_wavelength
    fac_time = width_time**2 * rs.C_0**2 / 2

    k_part = np.exp(-((k_list - k0) ** 2) * fac_time)
    lam_part = np.array([1,0])
    theta_part = np.exp( -(theta_list ** 2) / (2*width_theta**2) ) #/ np.sqrt(np.sin(theta_list))
    phi_part = np.exp(1j*phi_list)
    
    pulse_vals = np.einsum(
        "k,l,t,p->kltp",
        k_part,
        lam_part,
        theta_part,
        phi_part,
        optimize=True
    ) * norm_fac
    
    incident_pw = rs.WaveFunctionPlaneWave(k_list, theta_list, phi_list, pulse_vals)
    incident_pw.info['central_wavelength'] = central_wavelength

    return incident_pw

def get_truncated_theta_list(incident_rough):
    theta_rough = incident_rough.theta_list
    theta_range_indices = incident_rough._analyze_theta_content()
    
    theta_min = 0
    theta_max = theta_rough[theta_range_indices[-1]]
        
    return np.linspace(theta_min, theta_max, len(theta_rough))

def get_pulse_old(num_k, num_phi, num_theta, norm_fac, width_time, width_space, central_wavelength):
    '''Creates incident pulse for boosting article'''
    
    ### In PW basis
    k_list = get_k_list_for_pulse(num_k, width_time, central_wavelength)

    rough_theta_list = np.linspace(0, np.pi, num_theta) # use 0.4 for incident in lab
    phi_list = np.linspace(0, 2 * np.pi, num_phi)

    incident_rough_vals = rs.get_plane_pulse_wave_func_pw(
        k_list,
        rough_theta_list,
        phi_list,
        central_wavelength,
        width_time,
        width_space,
        norm_fac,
        ifPositiveHelicity=True,
    )
    incident_rough = rs.WaveFunctionPlaneWave(k_list, rough_theta_list, phi_list, incident_rough_vals)
    
    
    truncated_theta_list = get_truncated_theta_list(incident_rough)
    
    incident_precise_vals = rs.get_plane_pulse_wave_func_pw(
        k_list,
        truncated_theta_list,
        phi_list,
        central_wavelength,
        width_time,
        width_space,
        norm_fac,
        ifPositiveHelicity=True,
    )
    
    incident_precise = rs.WaveFunctionPlaneWave(k_list, truncated_theta_list, phi_list, incident_precise_vals)
    incident_precise.info['central_wavelength'] = central_wavelength

    return incident_precise
