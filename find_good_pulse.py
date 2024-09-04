"""
@author: Maxim Vavilin maxim@vavilin.de
"""
import os
import math
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import matplotlib as mpl
import treams
import copy
import time
import treams.special as sp

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

lam = 1

vec_z = get_polarization_vec(lam, [0], [0]).squeeze()
print('z-dir', vec_z)

# theta = 0.2
# phi = 0.1

# print('Vec rotated with wigner')
# vec1 = get_method_wigner
# print(vec1)

# print('Vec reference')

# thetas = np.linspace(0, np.pi, 100)

# wigner_vals = sp.wignersmalld(1, 1, 1, thetas)
# plt.plot(thetas, wigner_vals)
# plt.show()