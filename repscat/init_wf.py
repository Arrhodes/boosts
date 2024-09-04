# -*- coding: utf-8 -*-
"""
@author: Maxim Vavilin maxim@vavilin.de
"""


import numpy as np
import treams.special as sp
from repscat.constants import C_0

from scipy.integrate import quad  # , dblquad


def complex_quadrature(func, a, b, **kwargs):
    def real_func(x, *args):
        return np.real(func(x, *args))

    def imag_func(x, *args):
        return np.imag(func(x, *args))

    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (
        real_integral[0] + 1j * imag_integral[0],
        real_integral[1:],
        imag_integral[1:],
    )


def integrand_theta(theta, J, m, lam, k, Dr2):
    return (
        np.exp(-(k**2) * (1 - np.cos(theta)**2) * Dr2)
        * np.abs(np.cos(theta))
        * (1 + lam * np.cos(theta))
        * sp.wignersmalld(J, m, lam, theta)
    )



def get_plane_pulse_wave_func_pw(
    k_list,
    theta_list,
    phi_list,
    center_wavelength,
    dt,
    dr,
    mult_f,
    ifPositiveHelicity,
):
    #    Generates WF in PW basis
    k0 = 2 * np.pi / center_wavelength
    Dt2C2 = dt**2 * C_0**2 / 2
    Dr2 = dr**2 / 2

    exp_k = np.exp(-((k_list - k0) ** 2) * Dt2C2)
    exp_k_theta = np.zeros((len(k_list), len(theta_list)), dtype=complex)
    for i_k, k in np.ndenumerate(k_list):
        exp_k_theta[i_k, :] = np.exp(-(k**2) * (1 - np.cos(theta_list) ** 2) * Dr2)
    exp_phi = np.exp(1j * phi_list)
    fac_lam_theta = np.zeros((2, len(theta_list)), dtype=complex)
    for i_lam in [0, 1]:
        lam = 1 if i_lam == 0 else -1
        if not (ifPositiveHelicity and lam == -1):
            fac_lam_theta[i_lam, :] = np.abs(np.cos(theta_list)) * (
                1 + lam * np.cos(theta_list)
            )

    wave_func_pw = (
        np.einsum("b,bc,d,ac->bacd", exp_k, exp_k_theta, exp_phi, fac_lam_theta) * mult_f
    )
    # print("Generated WF in PW basis")
    return wave_func_pw


def get_plane_pulse_wave_func_am(
    k_list,
    N_J,
    center_wavelength,
    width_time,
    width_space,
    mult_f,
    ifPositiveHelicity,
    # ifPositiveEta,
):
    # Generate WF in AM basis. SOMETHING WRONG HERE, NEEDS TESTING

    k0 = 2 * np.pi / center_wavelength
    N_k = np.shape(k_list)[0]
    Max_theta = np.pi

    Dt2C2 = width_time**2 * C_0**2 / 2
    Dr2 = width_space**2 / 2

    wave_func_am = np.zeros((N_k, 2, N_J, 2 * N_J + 1), complex)
    lam_range = [1] if ifPositiveHelicity else [1, -1]
    # min_theta_param = 0 if ifPositiveEta else Min_eta
    for lam in lam_range:
        i_lam = 0 if lam == 1 else 1
        for i_k, k in np.ndenumerate(k_list):
            for J in range(1, N_J + 1):
                i_J = J - 1
                for m in range(-J, J + 1):
                    i_m = m + N_J
                    if m == 1:  ## Integration over phi gives initial polarization
                        cq = complex_quadrature(
                            integrand_theta,
                            0,
                            Max_theta,
                            args=(J, m, lam, k, Dr2),
                        )
                        int_theta = (
                            np.sqrt((2 * J + 1) / (4 * np.pi)) * cq[0] * (2 * np.pi)
                        )
                        wave_func_am[i_k, i_lam, i_J, i_m] = (
                            mult_f * np.exp(-((k - k0) ** 2) * Dt2C2) * int_theta
                        )
    return wave_func_am
