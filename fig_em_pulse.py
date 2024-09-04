"""
@author: Maxim Vavilin maxim@vavilin.de
"""
import matplotlib.pyplot as plt
import numpy as np
from wigners import wigner_3j
from repscat.fields.spherical_pulse import get_spherical_pulse_wavefunction
from repscat.constants import C_0, C_0_SI, EPSILON_0_SI, H_BAR_SI
from repscat.aux_funcs import add_colorbar_to_figure
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
import cmasher as cmr

def get_polarization_vec(lam, theta_list, phi_list):
    cos_theta = np.cos(theta_list)
    sin_theta = np.sin(theta_list)
    cos_phi = np.cos(phi_list)
    sin_phi = np.sin(phi_list)
    fac = -1 / np.sqrt(2)
    polariz_vec = np.zeros((3,len(theta_list), len(phi_list)), dtype=complex)
    for i_theta, _ in np.ndenumerate(theta_list):
            polariz_vec[0,i_theta,:] = fac * (lam * cos_theta[i_theta] * cos_phi - 1j * sin_phi)
            polariz_vec[1,i_theta,:] = fac * (lam * cos_theta[i_theta] * sin_phi + 1j * cos_phi)
            polariz_vec[2,i_theta,:] = fac * (-lam) * sin_theta[i_theta]

    return polariz_vec

def get_x_list(x_max, num):
    x_min = -x_max
    return np.linspace(x_min, x_max, num)

def get_z_list(z_max, num):
    z_min = -z_max
    return np.linspace(z_min, z_max, 4*num)

def get_dimensional_constant_in_SI():
    # Includes nm to m conversion
    riemann_silberstein_part = np.sqrt(C_0_SI*H_BAR_SI/EPSILON_0_SI)/np.sqrt((2*np.pi)**3)
    nm_to_m_square = 1e18
    return riemann_silberstein_part * nm_to_m_square

def get_riemann_silberstein(wavefunction, lam, time, x_list, z_list):
    k_list = wavefunction.k_list
    theta_list = wavefunction.theta_list
    phi_list = wavefunction.phi_list
    
    measure_k = np.diff(k_list, append=k_list[-1]) * k_list
    measure_theta = np.diff(theta_list, append=theta_list[-1]) * np.sin(theta_list)
    measure_phi = np.diff(phi_list, append=phi_list[-1])
    
    # Legend: x for coordinate x, k for wavenumber, t for theta, p for phi
    exponent_x = np.exp(
        np.einsum(
            'x,k,t,p->xktp',
            1j*x_list,
            k_list,
            np.sin(theta_list),
            np.cos(phi_list)
            ) 
    )
    exponent_z = np.exp(
        np.einsum(
            'z,k,t->zkt',
            1j*z_list,
            k_list,
            np.cos(theta_list)
            ) 
    )
    exponent_time = np.exp(-1j*C_0*time*k_list)
    
    polarization_vec = get_polarization_vec(lam, theta_list, phi_list)
    
    if lam == 1:
        i_lam = 0
    elif lam == -1:
        i_lam = 1
        
    dim_factor = get_dimensional_constant_in_SI()
    
    return np.einsum(
        'k,t,p,xktp,zkt,k,dtp,ktp->dxz',
        measure_k,
        measure_theta,
        measure_phi,
        exponent_x,
        exponent_z,
        exponent_time,
        polarization_vec,
        np.squeeze(wavefunction.vals[:,i_lam,:,:]),
        optimize=True
        ) * dim_factor
    
    
def get_energy_density(wavefunction, time, x_list, z_list):
    riemann_silberstein_plus = get_riemann_silberstein(wavefunction, 1, time, x_list, z_list)
    riemann_silberstein_minus = get_riemann_silberstein(wavefunction, -1, time, x_list, z_list)
    
    sum = riemann_silberstein_plus+np.conj(riemann_silberstein_minus)
    
    return np.einsum(
        'dxz,dxz->xz',
        sum,
        np.conj(sum),
        optimize=True
    ) * EPSILON_0_SI
    
def add_text_boxes_to_figure(axs, t_max):
    axs[1].text(0.96, 0.95, f'$t={-t_max}$ fs', transform=axs[0].transAxes, fontsize=12, 
            verticalalignment='top', horizontalalignment='right', color='white')
    axs[1].text(0.96, 0.95, f'$t={0}$ fs', transform=axs[1].transAxes, fontsize=12, 
            verticalalignment='top', horizontalalignment='right', color='white')
    axs[2].text(0.96, 0.95, f'$t={t_max}$ fs', transform=axs[2].transAxes, fontsize=12, 
            verticalalignment='top', horizontalalignment='right', color='white')
           

def plot(wavefunction, num_axis, ax=None):
    """ Plot for article: EM-field of the pulse 
    lengths are in nm, time in fs"""
    if ax is None:
        fig, axs = plt.subplots(3,1, dpi=300)

    cmap = mpl.colormaps['viridis']
    
    x_max = 2000
    z_max = 4000
    
    x_list = get_x_list(x_max, num_axis)
    z_list = get_z_list(z_max, num_axis)

    t_max = 10
    
    energy_density_top = get_energy_density(wavefunction, -t_max, x_list, z_list)
    energy_density_mid = get_energy_density(wavefunction, 0, x_list, z_list)
    energy_density_bot = get_energy_density(wavefunction, t_max, x_list, z_list)
    
    Z, X = np.meshgrid(z_list, x_list, indexing='ij')
    plot_vals_top = np.real(energy_density_top).T
    plot_vals_mid = np.real(energy_density_mid).T
    plot_vals_bot = np.real(energy_density_bot).T
    
    maxval = np.max(np.maximum(np.maximum(plot_vals_top, plot_vals_mid), plot_vals_bot))
    minval = 0

    norm = mpl.colors.Normalize(vmin=minval, vmax=maxval) 
    
    axs[0].pcolormesh(Z/1000, X/1000, plot_vals_top, shading='auto', cmap=cmap, norm=norm)
    axs[1].pcolormesh(Z/1000, X/1000, plot_vals_mid, shading='auto', cmap=cmap, norm=norm)
    axs[2].pcolormesh(Z/1000, X/1000, plot_vals_bot, shading='auto', cmap=cmap, norm=norm)
    
    add_text_boxes_to_figure(axs, t_max)
    
    add_colorbar_to_figure(fig, minval, maxval, cmap, 'Energy density, $J/\\text{m}^{3}$')
    
    for ax in axs:
        ax.set_ylabel('x, $\mu$m')
        ax.set_aspect('equal')
        ax.set
        
    axs[0].set_xticklabels([])    
    axs[1].set_xticklabels([])
    axs[2].set_xlabel('z, $\mu$m')
        

    
    plt.savefig(f"figs/Pulse.png", bbox_inches="tight")
    # plt.show()
    
    return ax

