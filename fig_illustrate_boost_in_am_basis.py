"""
@author: Maxim Vavilin maxim@vavilin.de
"""
import matplotlib.pyplot as plt
from wigners import wigner_3j
from repscat.fields.spherical_pulse import get_spherical_pulse_wavefunction
import numpy as np

def get_theta_list():
    return np.linspace(0, np.pi, 600)

def get_phi_list():
    return np.linspace(0, 2*np.pi, 200)

def plot_of(xi):
    ########### Plot for article: boosted angular momentum basis state
    jay_max = 5
    lam = 1
    jay = 2
    m = 0
    center_wavelength = 700 #2*np.pi/0.009
    width_time = 50
    norm_fac = 1
    num_k = 200
    

    rep = get_spherical_pulse_wavefunction(
        lam, jay, m, center_wavelength, width_time, norm_fac, num_k, jay_max
    )
    fig, ax = plt.subplots(nrows=1, ncols=1)
    
    ax.plot(rep.k_list*1000, rep.vals[:, 0, jay-1, m+jay_max], linestyle='--', label=f'$\\braket{{k{jay}{m}{lam}|f}}$', linewidth=1.5)

    theta_list = get_theta_list()
    phi_list = get_phi_list()
    rep_pw = rep.in_plane_wave_basis(theta_list, phi_list)
    rep_boosted_pw = rep_pw.boost(xi)
    boosted_rep = rep_boosted_pw.in_angular_momentum_basis(jay_max)
    
    for b_jay in range(1, jay_max+1):
        vals = np.real(boosted_rep.vals[:, 0, b_jay-1, m+jay_max])
        ax.plot(boosted_rep.k_list*1000, vals, linestyle='-',label=f"$\\braket{{k{b_jay}{m}{lam}|f\'}}$", linewidth=1) 
    ax.set_xlabel('Wavenumber $k$, $\\text{$\mu$m}^{-1}$')
    ax.set_ylabel(r'Wavefunction $f^\prime_{jm\lambda}(k)$, m')
    legend = ax.legend(frameon = 1)
    frame = legend.get_frame()
    frame.set_color('white')
    frame.set_alpha(1)
    
    ax.yaxis.grid(True, linestyle=':', color='black', linewidth=0.5)

    plt.savefig(f'figs/Boost_WF_XI={xi}.png', bbox_inches='tight')
    # plt.show()
    
def plot():
    for xi in [-0.05, 0.05]:
        plot_of(xi)

