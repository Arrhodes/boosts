"""
@author: Maxim Vavilin maxim@vavilin.de
"""


import numpy as np
import repscat as rs
import copy
import treams.special as sp
import matplotlib.pyplot as plt

##### AM
def stretch_jays(vals, max_jay_new):
    shape_old = np.shape(vals)
    max_jay_old = shape_old[2]
    if max_jay_old == max_jay_new:
        return vals
    vals_stretched = np.zeros(
        (shape_old[0], shape_old[1], max_jay_new, 2 * max_jay_new + 1), dtype=complex
    )
    for jay in range(1, max_jay_old + 1):
        i_jay = jay - 1
        for m in range(-jay, jay + 1):  # m in new array
            i_m_new = m + max_jay_new  # the main point of the function
            i_m_old = m + max_jay_old
            vals_stretched[:, :, i_jay, i_m_new] = vals[:, :, i_jay, i_m_old]
    return vals_stretched


def boost_vals_am(vals, k_list_boosted, k_list, max_jay_new, xi):
    vals = stretch_jays(vals, max_jay_new)
    jay_list = np.arange(1, max_jay_new + 1)

    vals_boosted = np.zeros(
        (len(k_list_boosted), 2, max_jay_new, 2 * max_jay_new + 1), dtype=complex
    )
    for i_k1, k1 in np.ndenumerate(k_list_boosted):
        min_k = max(k1 * np.exp(-xi), k_list[0])
        max_k = min(k1 * np.exp(xi), k_list[-1])
        integral_region = np.linspace(min_k, max_k, len(k_list))
        interp_vals = np.zeros(
            (len(k_list), 2, max_jay_new, 2 * max_jay_new + 1), dtype=complex
        )
        for i_lam, lam in np.ndenumerate(np.array([1, -1])):
            for i_jay, jay in np.ndenumerate(jay_list):
                for m in range(-jay, jay + 1):
                    i_m = m + max_jay_new
                    interp_vals[:, i_lam, i_jay, i_m] = np.reshape(
                        np.interp(
                            integral_region,
                            k_list,
                            np.real(np.squeeze(vals[:, i_lam, i_jay, i_m])),
                        )
                        + 1j
                        * np.interp(
                            integral_region,
                            k_list,
                            np.imag(np.squeeze(vals[:, i_lam, i_jay, i_m])),
                        ),
                        (len(k_list), 1),
                    )

        theta_1 = np.zeros_like(integral_region)
        theta_2 = np.zeros_like(integral_region)
        for i_k2, k2 in np.ndenumerate(integral_region):  # will sum over this freq
            ct1 = (k1 * np.cosh(xi) - k2) / (k1 * np.sinh(xi))
            ct2 = (k1 - k2 * np.cosh(xi)) / (k2 * np.sinh(xi))
            if ct1 < -1:
                ct1 = -1
            if ct1 > 1:
                ct1 = 1
            if ct2 < -1:
                ct2 = -1
            if ct2 > 1:
                ct2 = 1
            theta_1[i_k2] = np.arccos(ct1)
            theta_2[i_k2] = np.arccos(ct2)

        wig_1 = np.zeros(
            (
                len(jay_list),
                2 * max_jay_new + 1,
                2,
                len(integral_region),
            ),
            dtype=complex,
        )
        wig_2 = np.zeros(
            (
                len(jay_list),
                2 * max_jay_new + 1,
                2,
                len(integral_region),
            ),
            dtype=complex,
        )
        for i_lam, lam in np.ndenumerate(np.array([1, -1])):
            for i_jay, jay in np.ndenumerate(jay_list):
                fac_jay = np.sqrt(2 * jay + 1)
                for m in range(-jay, jay + 1):
                    i_m = m + max_jay_new
                    wig_1[i_jay, i_m, i_lam, :] = fac_jay * sp.wignersmalld(
                        jay, m, lam, theta_1
                    )
                    wig_2[i_jay, i_m, i_lam, :] = fac_jay * sp.wignersmalld(
                        jay, m, lam, theta_2
                    )

        integral_measure = np.diff(integral_region, append=integral_region[-1])

        vals_boosted[i_k1] = (
            np.einsum(
                "k,imak,jmak,kajm->aim",  # sum over k2=k,j2=j
                integral_measure,
                wig_1,
                wig_2,
                interp_vals,
                optimize=True,
            )
            * 0.5
            / np.sinh(xi)
            / k1
        )
    return vals_boosted



##### PW


def pw_boosted_domain(k_list, theta_list, xi):
    boosted_theta_list = np.zeros_like(theta_list)
    boosted_k_array = np.zeros((len(theta_list), len(k_list)))
    boosted_theta_list = np.arccos((np.cos(theta_list) + np.tanh(xi)) / (1 + np.cos(theta_list) * np.tanh(xi)))
    boosted_k_array = np.einsum("k,e->ek", k_list, np.cosh(xi) + np.cos(theta_list) * np.sinh(xi))

    return boosted_k_array, boosted_theta_list


def pw_interpolated_vals(
    vals, uniform_k_list, boosted_k_array, boosted_theta_list, phi_list
):
    interp_vals = np.empty(vals.shape, dtype=complex)
    for i_lam in [0, 1]:
        for i_phi in range(len(phi_list)):
            for i_theta in range(len(boosted_theta_list)):
                interp_vals[:, i_lam, i_theta, i_phi] = np.interp(
                    uniform_k_list,
                    boosted_k_array[i_theta, :],
                    np.real(vals[:, i_lam, i_theta, i_phi]),
                ) + 1j * np.interp(
                    uniform_k_list,
                    boosted_k_array[i_theta, :],
                    np.imag(vals[:, i_lam, i_theta, i_phi]),
                )
                # if i_phi == 30 and i_eta==len(boosted_eta_list)-20 and i_lam==0:
                # plt.plot(boosted_k_array[i_eta, :], vals[:, i_lam, i_eta, i_phi], linestyle='--')
                # plt.plot(uniform_k_list, interp_vals[:, i_lam, i_eta, i_phi])
                # plt.plot(boosted_eta_list[-100:], vals[100, i_lam, -100:, i_phi], linestyle='--')
                # plt.plot(boosted_eta_list[-100:], interp_vals[100, i_lam, -100:, i_phi])
                # plt.show()
    # plt.plot(boosted_eta_list[-10:], vals[100, 0, -10:, 0], linestyle='--')
    # plt.plot(boosted_eta_list[-100:], interp_vals[100, 0, -100:, 0])
    # plt.show()
    return interp_vals
