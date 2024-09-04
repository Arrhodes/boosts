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
from wigners import wigner_3j


import repscat as rs
from repscat.fields.spherical_pulse import get_spherical_pulse_wavefunction

from incident_pulse import get_k_list_for_pulse, get_pulse


plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "Times",
        # "font.monospace": "Computer Modern Typewriter",
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.size": 15,
        "figure.dpi": 300,
    }
)
# plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams.update(
    {
        "font.family": "serif",
        # "font.monospace": "Computer Modern Typewriter",
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{libertine}\usepackage[libertine]{newtxmath}\usepackage{braket}\usepackage{siunitx}",
        "font.size": 15,
        "figure.dpi": 300,
    }
)

### Params for plot
norm_fac = 2.095e11
center_wavelength = 700  # Must be 700! and at least 10 fs
width_time = 20
num_k = 400
k_list = get_k_list_for_pulse(num_k, width_time, center_wavelength)

# print('kmin kmax', k_list[0], k_list[-1])

k_full_min = k_list[0]*np.exp(-np.abs(1.1))
k_full_max = k_list[-1]*np.exp(np.abs(1.1))
k_list_full = np.linspace(k_full_min, k_full_max, 200)

jay_max = 4 # must be 5 (try 3)
_, incident_pw = get_pulse(k_list, jay_max, norm_fac, width_time, center_wavelength)
incident_am = incident_pw.in_angular_momentum_basis(jay_max)

radius = 100

# ### Diag value for rest
# tmat_diag = rs.get_silicon_tmat(radius, jay_max, k_list)
# quantities_diag = tmat_diag.compute_transfer(incident_am)

# print('Quantities diagonal:')
# print(quantities_diag)



# incident_am.plot()
# plt.show()

# print('Photons = ', incident_pw.photons())

# boosted = incident_pw.boost(-1.1)

# print('Photons boosted= ', boosted.photons())


########### Plot for article: boosted wavefunction
# lam = 1
# jay = 1
# m = 1
# center_wavelength = 700
# width_time = 50
# norm_fac = 1
# num_k = 200
# jay_max = 6

# rep = get_spherical_pulse_wavefunction(
#     lam, jay, m, center_wavelength, width_time, norm_fac, num_k, jay_max
# )
# # print(rep.k_list[0], rep.k_list[-1])
# plt.plot(rep.k_list*1000, rep.vals[:, 0, jay-1, m+jay_max], linestyle='--', label='$\\braket{k111|f}$')

# xi=-0.05
# boosted_rep = rep.boost(xi)
# for b_jay in range(1, jay_max+1):
#     plt.plot(boosted_rep.k_list*1000, boosted_rep.vals[:, 0, b_jay-1, m+jay_max], linestyle='-',label=f"$\\braket{{k{b_jay}11|f\'}}$") 
# plt.xlabel('Wavenumber $k$, $\\text{$\mu$m}^{-1}$')
# plt.ylabel(r'Wavefunction $f_{jm\lambda}(k)$')
# legend = plt.legend(frameon = 1)
# frame = legend.get_frame()
# frame.set_color('white')
# frame.set_alpha(1)

# plt.savefig(f'Boost_WF_XI={xi}.png', bbox_inches='tight')
# # plt.show()

############################ Plot for article : j_max of T-matrix where to truncate
# radius = 100

# for jay_max in range(1, 4 + 1):
#     tmat = rs.get_silicon_tmat(radius, jay_max, k_list)
#     max_str = "\\text{max}"
#     tmat.plot_norm(label=f"$j_{max_str}=$ {jay_max}", linestyle="-")

# plt.xlabel("Wavenumber $k$, $\\text{$\mu$m}^{-1}$")
# plt.ylabel("Interaction cross-section $||T(k)||^2$")

# legend = plt.legend(frameon=1)
# frame = legend.get_frame()
# frame.set_color("white")
# frame.set_alpha(1)

# # plt.show()
# plt.savefig(f"InteractionCS1.png", bbox_inches='tight')


############################ Plot for article : Silicon properties at full wavelength
# plt.rcParams.update(
#     {
#         "font.size": 18,
#     }
# )
# silicon = rs.Silicon()
# silicon.plot(k_list_full)

# plt.xlabel("Wavenumber $k$, $\\text{$\mu$m}^{-1}$")

# legend = plt.legend(frameon=1)
# frame = legend.get_frame()
# frame.set_color("white")
# frame.set_alpha(1)

# # plt.show()
# plt.savefig(f"Silicon_wide.png", bbox_inches='tight')

########################### Plot for article : j_max of T-matrix at full wavelength
# plt.rcParams.update(
#     {
#         "font.size": 18,
#     }
# )
# radius = 100

# for jay_max in range(1, 5 + 1):
#     tmat = rs.get_silicon_tmat(radius, jay_max, k_list_full)
#     max_str = "\\text{max}"
#     tmat.plot_norm(label=f"$j_{max_str}=$ {jay_max}", linestyle="-")

# plt.xlabel("Wavenumber $k$, $\\text{$\mu$m}^{-1}$")
# plt.ylabel("Interaction cross-section $||T(k)||^2$")

# legend = plt.legend(frameon=1)
# frame = legend.get_frame()
# frame.set_color("white")
# frame.set_alpha(1)

# # plt.show()
# plt.savefig(f"InteractionCS_total.png", bbox_inches='tight')

############# Plot for article: Interaction cross-section as function of v/c
# plt.rcParams.update(
#     {
#         "font.size": 18,
#     }
# )
# radius = 100

# for jay_max in range(1, 5 + 1):
#     tmat = rs.get_silicon_tmat(radius, jay_max, k_list_full)
#     max_str = "\\text{max}"
#     tmat.plot_norm(label=f"$j_{max_str}=$ {jay_max}", linestyle="-")

# plt.xlabel("Wavenumber $k$, $\\text{$\mu$m}^{-1}$")
# plt.ylabel("Interaction cross-section $||T(k)||^2$")

# legend = plt.legend(frameon=1)
# frame = legend.get_frame()
# frame.set_color("white")
# frame.set_alpha(1)

# # plt.show()
# plt.savefig(f"InteractionCS_total.png", bbox_inches='tight')


############# Plot for article: Transfer in object's frame (use test_scatter_obj_vs_lab instead!)
# plt.rcParams.update(
#     {
#         "font.size": 18,
#     }
# )

# momentum_list = []
# energy_list = []
# num_xi = 300
# xi_list = np.linspace(-1.1, 1.1, num_xi, endpoint=True) # boost of object

# for xi in xi_list:
#     print('xi = ', xi)
#     minus_xi = -xi # fields are reversely boosted
#     boosted_pw = incident_pw.boost(minus_xi)
#     boosted_inc = boosted_pw.in_angular_momentum_basis(jay_max)

#     k_list_boosted = boosted_inc.k_list

#     tmat = rs.get_silicon_tmat(radius, jay_max, k_list_boosted)

#     quantities =  tmat.compute_transfer(boosted_inc)

#     energy = quantities['energy']
#     momentum_z = quantities['momentum_z']

#     Enew = np.cosh(xi)*energy +np.sinh(xi)*momentum_z*rs.C_0_SI
#     Pnew = np.sinh(xi)*energy/rs.C_0_SI + np.cosh(xi)*momentum_z

#     energy_list.append(energy)
#     momentum_list.append(momentum_z)

# # E = energy_list[0]
# # P = momentum_list[0]

# # xi = xi_list[0]

# # print(f'In object frame:')
# # print(f'E={np.real(E)}, P={np.real(P)}')

# # Enew = np.cosh(xi)*E +np.sinh(xi)*P*C_0_SI
# # Pnew = np.sinh(xi)*E/C_0_SI + np.cosh(xi)*P

# # print([Enew, Pnew])

# fig, ax = plt.subplots( nrows=1, ncols=1 ) #E/(hc)  = k
# ax.plot(np.tanh(xi_list),  np.real(momentum_list))
# ax.set_xlabel('v/c')
# ax.set_ylabel('Momentum in z transfer, $\\si{\kilogram\metre\per\second}$')
# fig.savefig(f'Momentum_transfer_obj_frame_{len(xi_list)}.png', bbox_inches='tight')

# fig, ax = plt.subplots( nrows=1, ncols=1 )
# ax.plot(np.tanh(xi_list),  1e3*np.real(energy_list)) # to have mJ
# ax.set_xlabel('v/c')
# ax.set_ylabel('Energy transfer, $\\si{\milli\joule}$')
# fig.savefig(f'Energy_transfer_obj_frame_{len(xi_list)}.png', bbox_inches='tight')


######################## Reference for object boost: Transfer in object's frame

# xi_ref = -0.000002 # boost of field

# print('xi = ', xi_ref)
# boosted_pw = incident_pw.boost( xi_ref)
# boosted_am = boosted_pw.in_angular_momentum_basis(jay_max)
# # boosted_am = rs.boost_pw_representation(boosted_pw, max_jay)

# k_list_boosted = boosted_am.k_list
# tmat = rs.get_silicon_tmat(radius, jay_max, k_list_boosted)
# num_k = len(k_list_boosted)

# # scattered_am = tmat.scatter(boosted_am)
# # momentum = tmat.compute_transfer(incident,'momentum_z')
# # energy = tmat.compute_transfer(incident,'energy')
# quantities =  tmat.compute_transfer(boosted_am)

# E = quantities['energy']
# P = quantities['momentum_z']

# # xi = xi_list[0]

# # print(f'In object frame:')
# # print(f'E={np.real(E)}, P={np.real(P)}')

# Enew = np.cosh(-xi_ref)*E +np.sinh(-xi_ref)*P*rs.C_0_SI
# Pnew = np.sinh(-xi_ref)*E/rs.C_0_SI + np.cosh(-xi_ref)*P

# print('Energy', Enew)
# print('Momentum', Pnew)


###########################################################

# ### Silicon data
# silicon = rs.Silicon()
# silicon.plot()
# plt.show()

# ### Incident pulse basis
# xi = 0.01
# num_k = 200 #200
# jay_max = 1
# radius = 100


# k2_list = get_k2_list(num_k)
# k_list = rs.boost_wavenumbers(k2_list, xi)
# k1_list = rs.boost_wavenumbers(k_list, xi)

# # incident_am, incident_pw = get_pulse(k2_list, jay_max)
# incident_am, incident_pw = get_pulse_small_jay(k2_list, jay_max)

# ## Check if goes to initial
# incident_pw_boosted = incident_pw.boost(-xi) # Minus here
# # incident_pw_boosted.plot()
# # incident_pw.plot()
# # plt.show()

# # incident_am.plot()
# # plt.show()

# # incident_pw.plot()
# # plt.show()

# # analyze_quantities(incident_pw, xi)

# transfer_no_boost(incident_am, radius)

# E1, P1 = transfer_boosted_in_lab_frame(incident_am, radius, xi)

# # incident_pw.plot()
# # plt.show()
# E2, P2 = transfer_boosted_object_frame(incident_pw, jay_max, radius, xi)

# print(rs.boost_E_Pz(E1,P1,xi))
# print(rs.boost_E_Pz(E2,P2,xi))


#### Correct results for boost (+)
# num_k = 150 #200
# jay_max = 5
# k2_list = rs.get_k2_list(num_k)
# incident_am, incident_pw = get_pulse(k2_list, jay_max)

# radius = 100
# momentum_list = []
# energy_list = []
# num_xi = 300
# xi_list = np.linspace(-1.12, 1.12, num_xi, endpoint=True)
# # xi_list = np.array([-0.001])
# for xi in xi_list:
#     print('xi = ', xi)
#     boosted_pw = incident_pw.boost( xi)
#     boosted_am = boosted_pw.in_angular_momentum_basis(jay_max)
#     # boosted_am = rs.boost_pw_representation(boosted_pw, max_jay)

#     k_list_boosted = boosted_am.k_list
#     tmat = rs.get_silicon_tmat(radius, jay_max, k_list_boosted)
#     num_k = len(k_list_boosted)

#     # scattered_am = tmat.scatter(boosted_am)
#     # momentum = tmat.compute_transfer(incident,'momentum_z')
#     # energy = tmat.compute_transfer(incident,'energy')
#     quantities =  tmat.compute_transfer(boosted_am)

#     momentum_list.append(quantities['momentum_z'])
#     energy_list.append(quantities['energy'])

# momentum_list.reverse()
# energy_list.reverse()

# # E = energy_list[0]
# # P = momentum_list[0]

# # xi = xi_list[0]

# # print(f'In object frame:')
# # print(f'E={np.real(E)}, P={np.real(P)}')

# # Enew = np.cosh(xi)*E +np.sinh(xi)*P*C_0_SI
# # Pnew = np.sinh(xi)*E/C_0_SI + np.cosh(xi)*P

# # print([Enew, Pnew])

# fig, ax = plt.subplots( nrows=1, ncols=1 ) #E/(hc)  = k
# ax.plot(np.tanh(xi_list),  np.real(momentum_list))
# ax.set_xlabel('v/c')
# ax.set_ylabel('Momentum in z transfer')
# fig.savefig(f'NEWESTPW_momentum_transfer_plane_{len(xi_list)}.png')

# fig, ax = plt.subplots( nrows=1, ncols=1 )
# ax.plot(np.tanh(xi_list),  np.real(energy_list))
# ax.set_xlabel('v/c')
# ax.set_ylabel('Energy transfer')
# fig.savefig(f'NEWESTPW_energy_transfer_plane_{len(xi_list)}.png')


###########       C O M P U T E     T R A N S F E R    I N     L A B F R A M E
# k2_list = incident.domain["k_list"] # incident freq
# radius = 100

# xi_list = np.array([-xi])
# k_lists = np.zeros((len(xi_list), num_k)) # Tmat silicon freq
# k1_lists = np.zeros((len(xi_list), num_k)) # Scattered freq
# for i_xi, xi in enumerate(xi_list):
#     k_lists[i_xi] = np.linspace(k2_list[0]*np.exp(-np.abs(xi)), k2_list[-1]*np.exp(np.abs(xi)), num_k) # diag tmat freq
#     k1_lists[i_xi] = np.linspace(k2_list[0]*np.exp(-2*np.abs(xi)), k2_list[-1]*np.exp(2*np.abs(xi)), num_k) # scat tmat freq

#     tmat = rs.get_silicon_tmat(radius, max_jay, np.squeeze(k_lists[i_xi]))

#     t1 = time.time()
#     tmat_boosted = tmat.boost_tmat(xi)  # (k1,k2,lam1,lam2,j1,j2,m1,m2)
#     print('Time for boosting: ', - t1 + time.time())

# ############         P L O T   T - M A T R I X   E L E M E N T
# lam1 = 1
# lam2 = 1
# jay1 = 1
# jay2 = 1
# m1 = 1
# m2 = 1
# tmat_boosted.plot(lam1,lam2,jay1,jay2,m1,m2)
# plt.show()

# ###########       C O M P U T E     T R A N S F E R    I N     L A B F R A M E
# # incident.plot()
# # plt.show()
# print(f"E={np.real(tmat_boosted.compute_transfer(incident,'energy'))}")
# print(f"P={np.real(tmat_boosted.compute_transfer(incident,'momentum_z'))}")

######################################################################################


# #####################   M A I N   C O M M A N D S  ###############################
# quantities, photon_density_list = get_quantity(boosted_pw, normalize=False)
# print(quantities)
# radius = 100
# tmat = get_silicon_tmat(radius, max_J, k_list, center_wavelength)
# scattered_am = scatter_representation(tmat, incident_am)
# transfer = get_transfer(tmat, incident_am, 'energy')
