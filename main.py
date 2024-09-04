"""
@author: Maxim Vavilin maxim@vavilin.de
"""
import numpy as np
import matplotlib.pyplot as plt
import fig_momentum_transfer_via_boosted_tmat, fig_poly_tmatrix_element, fig_em_pulse, \
    fig_tmatrix_jay_max, fig_silicon_properties, fig_illustrate_boost_in_am_basis, \
    fig_transfer_both_frames, fig_angular_content, compare_two_transfer_methods, boost_precisely
from incident_pulse import get_pulse_old, get_pulse
import time

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": (
            r"\usepackage{amsmath}"
            + r"\usepackage{libertine}"
            + r"\usepackage[libertine]{newtxmath}"
            + r"\usepackage{braket}"
            + r"\usepackage{siunitx}"
        ),
        "font.size": 15,
        "figure.dpi": 300,
    }
)

def initialize_pulse(num_k):
    """Incident pulse in lab frame, units are in femtoseconds and nanometers.
    Code is tuned for pulse with central wavelength 700 nm and at least 10 fs width
    """
    norm_fac = 3.25e11 # has units of nm
    central_wavelength = 700
    width_time = 10 
    width_theta = 0.1 # 
    num_phi = 75 # 75
    num_theta = 150 # 150 
    # num_k = 200  # 150 for lab transfer tmat, 200 for other

    incident_pw = get_pulse(
        num_k,
        num_phi,
        num_theta,
        norm_fac,
        width_time,
        width_theta,
        central_wavelength,
    )
    return incident_pw

def plot_all_plots():
    # * Initialize incident bullet-like pulse
    incident_pw = initialize_pulse(200) # 200

    # * Plot transfer in both frames (with cross sec of absorbtion)
    xi_max = 1.1
    len_xi = 400 #400 
    jay_max = 5 #5
    
    # * Plot silicon and Tmat properties
    fig_tmatrix_jay_max.plot('silicon', 5, incident_pw.k_list, xi_max)
    fig_tmatrix_jay_max.plot('gold_cluster', 5, incident_pw.k_list, xi_max)
    # fig_tmatrix_jay_max.plot('gold_sphere', 5, incident_pw.k_list, xi_max)
    
    fig_silicon_properties.plot(incident_pw.k_list, xi_max, incident_pw)
    
    # * Illustrate boost in AM basis
    fig_illustrate_boost_in_am_basis.plot()

    # * Plot transfer via freq-diagonal Tmat for silicon sphere and 4 golden spheres 
    xi_list = np.linspace(-xi_max, xi_max, len_xi)
    fig_transfer_both_frames.plot('silicon', incident_pw, jay_max, xi_list, load = False)
    fig_transfer_both_frames.plot('gold_cluster', incident_pw, jay_max, xi_list, load = False)
    # fig_transfer_both_frames.plot('gold_sphere', incident_pw, jay_max, xi_list, load = False)
    
    # * Plot angular photon density of pulse
    # incident_pw.plot_angular_photon_density()
    # plt.show()
    
    # # * Plot how the pulse looks like
    num_axis = 50
    fig_em_pulse.plot(incident_pw, num_axis)
    
    # # * Plot boosted tmatrix element
    radius = 150 # do not change
    fig_poly_tmatrix_element.plot(radius, incident_pw.k_list) 
   
    # # * Plot momentum transfer via boosted tmatrix
    # incident_pw_for_tmat = initialize_pulse(150) # 150
    # jay_max_for_lab_frame_transfer = 6
    # fig_momentum_transfer_via_boosted_tmat.plot(incident_pw_for_tmat, jay_max_for_lab_frame_transfer, radius)
    

if __name__ == "__main__":
    plot_all_plots()
