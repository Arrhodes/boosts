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

from repscat.tmatrix_diag import TmatrixDiagonal
from incident_pulse import get_k_list_for_pulse, get_pulse
from repscat.boost_tmat_at_k1 import get_k2_small_list


plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        # "font.monospace": "Computer Modern Typewriter",
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{libertine}\usepackage[libertine]{newtxmath}\usepackage{braket}\usepackage{siunitx}",
        "font.size": 15,
        "figure.dpi": 300,
    }
)

# from boost_tmat_precise_pw_final import get_k2_list_final

t1 = time.time()
###### Test parameters
norm_fac = 2.095e11
center_wavelength = 700  # Must be 700! and at least 10 fs
width_time = 20
num_k = 150 # do 100
jay_max = 3
k2_list = get_k_list_for_pulse(num_k, width_time, center_wavelength)

####### Set diagonal T-matrix
### Silicon

radius = 100


##### initialize field and scatter in obj

_, incident_pw = get_pulse(k2_list, jay_max, norm_fac, width_time, center_wavelength)
incident_am = incident_pw.in_angular_momentum_basis(jay_max)
# incident_pw = incident_am.in_plane_wave_basis(incident_pw.theta_list, incident_pw.phi_list)
# incident_am = incident_pw.in_angular_momentum_basis(jay_max)

momentum_list = []
energy_list = []
num_xi = 27
xi_list = np.linspace(-0.015, 0.015, num_xi, endpoint=True)  # boost of object

energy_obj_list = []
momentum_obj_list = []

for xi in xi_list:
    print("xi = ", xi)
    inc_pw_obj = incident_pw.boost(-xi)
    inc_am_obj = inc_pw_obj.in_angular_momentum_basis(jay_max)

    tmat_diag_obj = rs.get_silicon_tmat(radius, jay_max, inc_am_obj.k_list)

    scat_am_obj = tmat_diag_obj.scatter(inc_am_obj)

    scat_pw_obj = scat_am_obj.in_plane_wave_basis(
        inc_pw_obj.theta_list, inc_pw_obj.phi_list
    )
    scat_pw_lab = scat_pw_obj.boost(xi)

    ## obj frame
    out_pw_obj = inc_pw_obj + scat_pw_obj

    mom_diff_obj = inc_pw_obj.momentum_z() - out_pw_obj.momentum_z()
    energy_diff_obj = inc_pw_obj.energy() - out_pw_obj.energy()

    energy_obj_list.append(energy_diff_obj)
    momentum_obj_list.append(mom_diff_obj)

    ### lab frame
    energy_diff_lab = np.cosh(xi)*energy_diff_obj + np.sinh(-xi)*mom_diff_obj*rs.C_0_SI
    mom_diff_lab = np.sinh(-xi)*energy_diff_obj/rs.C_0_SI + np.cosh(xi)*mom_diff_obj

    # mom_diff_lab = incident_pw.momentum_z() - out_pw_lab.momentum_z()
    # energy_diff_lab = incident_pw.energy() - out_pw_lab.energy()

    energy_list.append(energy_diff_lab)
    momentum_list.append(mom_diff_lab)

    if xi == 0.01:
        print('Obj frame:')
        print('Energy ', energy_diff_obj)
        print('Momentum ', mom_diff_obj)

        print('Lab frame:')
        print('Energy ', energy_diff_lab)
        print('Momentum ', mom_diff_lab)


## several points from lab frame poly tmat
xi_list_poly_tmat = np.array([0.005, 0.01])
momentum_list_poly_tmat = []
energy_list_poly_tmat = []
for xi_poly_tmat in xi_list_poly_tmat:
    k_min = k2_list[0] * np.exp(-np.abs(xi_poly_tmat))
    k_max = k2_list[-1] * np.exp(np.abs(xi_poly_tmat))
    k_list = np.linspace(k_min, k_max, num_k)

    tmat_diag_for_poly = rs.get_silicon_tmat(radius, jay_max, k_list)

    k1_list = np.linspace(
        k_list[0] * np.exp(-np.abs(xi_poly_tmat)),
        k_list[-1] * np.exp(np.abs(xi_poly_tmat)),
        num_k,
    )
    tmat_boosted = tmat_diag_for_poly.boost_precise(xi_poly_tmat, k1_list, num_k)
    quantities_poly_tmat =  tmat_boosted.compute_transfer(incident_am)
    print(quantities_poly_tmat)
    energy_list_poly_tmat.append(quantities_poly_tmat['energy'])
    momentum_list_poly_tmat.append(quantities_poly_tmat['momentum_z'])

# momentum_list_poly_tmat = [
#     4.656268606086651e-13,
#     4.656272320877665e-13,
#     4.655519153966842e-13,
#     4.655464896219911e-13,
#     4.655409778891644e-13
# ]

# energy_list_poly_tmat = [
#     1.418260215009634e-06,
#     1.417873031968526e-06,
#     3.5082107594924263e-06,
#     3.5103021851776227e-06,
#     3.5124254515891636e-06
# ]


fig, ax = plt.subplots(nrows=1, ncols=1)  # E/(hc)  = k
ax.plot(np.tanh(xi_list), np.real(momentum_list)) # lab frame from pw
ax.scatter(np.tanh(xi_list_poly_tmat), np.real(momentum_list_poly_tmat))
ax.set_xlabel("v/c")
ax.set_ylabel("Momentum in z transfer, $\\si{\kilogram\metre\per\second}$")
fig.savefig(f"A5Momentum_transfer_LAB_frame_{len(xi_list)}_r_{radius}.png", bbox_inches="tight")

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(np.tanh(xi_list), 1e3 * np.real(energy_list))  # to have mJ
ax.scatter(np.tanh(xi_list_poly_tmat), 1e3 * np.real(energy_list_poly_tmat))
ax.set_xlabel("v/c")
ax.set_ylabel("Energy transfer, $\\si{\milli\joule}$")
fig.savefig(f"A5Energy_transfer_LAB_frame_{len(xi_list)}_r_{radius}.png", bbox_inches="tight")

plt.clf()

fig, ax = plt.subplots(nrows=1, ncols=1)  # E/(hc)  = k
# ax.plot(np.tanh(xi_list), np.real(momentum_list)) # lab frame from pw
ax.plot(np.tanh(xi_list), np.real(momentum_obj_list)) # obj frame from pw

ax.set_xlabel("v/c")
ax.set_ylabel("Momentum in z transfer, $\\si{\kilogram\metre\per\second}$")
fig.savefig(f"A6Momentum_transfer_OBJ_frame_{len(xi_list)}_r_{radius}.png", bbox_inches="tight")

fig, ax = plt.subplots(nrows=1, ncols=1)
# ax.plot(np.tanh(xi_list), 1e3 * np.real(energy_list))  # to have mJ
ax.plot(np.tanh(xi_list), 1e3 * np.real(energy_obj_list)) # obj frame from pw

ax.set_xlabel("v/c")
ax.set_ylabel("Energy transfer, $\\si{\milli\joule}$")
fig.savefig(f"A6Energy_transfer_OBJ_frame_{len(xi_list)}_r_{radius}.png", bbox_inches="tight")


# incident_am.plot()
# scattered.plot("-")
# plt.show()

# (incident_am + scattered).plot('-')
# plt.show()
# print('xi = ', xi)
# print( 'incident+scattered momentum_z  = ', (incident_am+scattered).momentum_z())
# print( 'incident+scattered energy  = ', (incident_am+scattered).energy())

# print( 'incident momentum_z  = ', (incident_am).momentum_z())
# print( 'incident_ energy  = ', (incident_am).energy())

# quantities =  tmat_diag.compute_transfer(incident_am)
# E = quantities['energy']
# P = quantities['momentum_z']
# print('Diag obj:')
# print('Energy', E)
# print('Momentum', P)

# ############################################## compute transfer

# len_k1 = num_k
# k1_list = np.linspace(
#     k_list[0] * np.exp(-np.abs(xi)), k_list[-1] * np.exp(np.abs(xi)), len_k1
# )
# scattered_vals = np.zeros((len_k1, 2, jay_max, 2*jay_max+1),dtype=complex)

# tmat_boosted = tmat_diag.boost_precise(xi, k1_list, len_k2)
# scattered = tmat_boosted.scatter(incident_am)

# # scattered.plot(':')
# # plt.show()

# incident_interp = incident_am.interpolate(scattered.k_list)
# (incident_interp+scattered).plot(':')
# print( 'incident_interp+scattered interp momentum_z = ', (incident_interp+scattered).momentum_z())
# print( 'incident_interp+scattered interp energy = ', (incident_interp+scattered).energy())
# print( 'scattered interp momentum_z = ', (incident_interp).momentum_z())
# print( 'scattered interp energy = ', (incident_interp).energy())

# plt.show()

# quantities =  tmat_boosted.compute_transfer(incident_am)
# E = quantities['energy']
# P = quantities['momentum_z']
# print('Boosted obj:')
# print('Energy', E)
# print('Momentum', P)

# # len_k1 = len_k2

# # k1_list = np.linspace(k2_list[0]*np.exp(-2*np.abs(xi)), k2_list[-1]*np.exp(2*np.abs(xi)), len_k1)
# # idx_k1 = np.abs(k1_list - 0.0085).argmin()
# # k1 = k1_list[idx_k1]


print("time elapsed: ", time.time() - t1)
