# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:47:19 2023

@author: Maxim Vavilin maxim@vavilin.de
"""
import matplotlib as mpl
import numpy as np
from scipy.integrate import quad, dblquad
import treams.special as sp
from wigners import wigner_3j, clebsch_gordan
from scipy.special import spherical_jn, spherical_yn
from math import isclose as isclose
import matplotlib.pyplot as plt
import h5py
from repscat.constants import C_0, C_0_SI, H_BAR_SI
import copy
from scipy import linalg as la
import sys


def add_colorbar_to_figure(fig, minval, maxval, cmap, text):
    fig.subplots_adjust(
            bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.1, hspace=0.1
        )
    cb_ax = fig.add_axes([0.67, 0.1, 0.02, 0.8])

    norm = mpl.colors.Normalize(vmin=minval, vmax=maxval)
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cb_ax,
        orientation="vertical",
        label=text,
    )

def interpolate_tmat(k3_list, k_list, tmat_diag_vals):
    jay1_max = tmat_diag_vals.shape[3]
    jay2_max = tmat_diag_vals.shape[4]

    tmat_interpol = np.zeros(
        (
            len(k3_list),
            2,
            2,
            jay1_max,
            jay2_max,
            2 * jay1_max + 1,
            2 * jay2_max + 1,
        ),
        dtype=complex,
    )

    for i_jay1 in range(jay1_max):
        jay1 = i_jay1 + 1
        for m1 in range(-jay1, jay1 + 1):
            i_m1 = m1 + jay1_max
            for i_jay2 in range(jay2_max):
                jay2 = i_jay2 + 1
                for m2 in range(-jay2, jay2 + 1):
                    i_m2 = m2 + jay2_max
                    for i_lam1 in [0, 1]:
                        for i_lam2 in [0, 1]:
                            real_part = np.interp(
                                k3_list,
                                k_list,
                                np.real(
                                    np.squeeze(
                                        tmat_diag_vals[
                                            :,
                                            i_lam1,
                                            i_lam2,
                                            i_jay1,
                                            i_jay2,
                                            i_m1,
                                            i_m2,
                                        ]
                                    )
                                ),
                            )
                            imag_part = np.interp(
                                k3_list,
                                k_list,
                                np.imag(
                                    np.squeeze(
                                        tmat_diag_vals[
                                            :,
                                            i_lam1,
                                            i_lam2,
                                            i_jay1,
                                            i_jay2,
                                            i_m1,
                                            i_m2,
                                        ]
                                    )
                                ),
                            )

                            tmat_interpol[
                                :,
                                i_lam1,
                                i_lam2,
                                i_jay1,
                                i_jay2,
                                i_m1,
                                i_m2,
                            ] = (
                                real_part + 1j * imag_part
                            )

    return tmat_interpol

def get_indices(jay_max, *indices):
        i_lam1 = 0 if indices[0] == 1 else 1
        i_lam2 = 0 if indices[1] == 1 else 1
        i_jay1 = indices[2] - 1
        i_jay2 = indices[3] - 1
        i_m1 = indices[4] + jay_max
        i_m2 = indices[5] + jay_max
        return i_lam1, i_lam2, i_jay1, i_jay2, i_m1, i_m2

def boost_E_Pz(E,Pz,xi):
        new_E = E*np.cosh(xi) + C_0_SI*Pz*np.sinh(xi)
        new_Pz = E*np.sinh(xi)/C_0_SI + Pz*np.cosh(xi)
        return new_E, new_Pz

def boost_wavenumbers(k_list, xi, num_k=None):
    k_min = k_list[0]*np.exp(-np.abs(xi))
    k_max = k_list[-1]*np.exp(np.abs(xi))
    num_k = len(k_list) if num_k is None else num_k
    return np.linspace(k_min,k_max, num_k)

def get_common_domain(k1_list, k2_list):

    min_k = min(k1_list[0], k2_list[0])
    max_k = max(k1_list[-1], k2_list[-1])
    num_k = max(len(k1_list), len(k2_list))

    return np.linspace(min_k, max_k, num_k)


def unwrap_tmat(tmat, num_k, max_J):
    if num_k == 1:
        tmat_unwrapped = np.zeros(
            (2, max_J, 2 * max_J + 1, 2, max_J, 2 * max_J + 1), dtype=complex
        )
        for i_lam_1 in [0, 1]:
            for i_lam_2 in [0, 1]:
                idx_1 = 0
                for J_1 in range(1, max_J + 1):
                    i_J_1 = J_1 - 1
                    for m_1 in range(-J_1, J_1 + 1):
                        i_m_1 = m_1 + max_J
                        idx_2 = 0
                        for J_2 in range(1, max_J + 1):
                            i_J_2 = J_2 - 1
                            for m_2 in range(-J_2, J_2 + 1):
                                i_m_2 = m_2 + max_J
                                tmat_unwrapped[
                                    i_lam_1, i_J_1, i_m_1, i_lam_2, i_J_2, i_m_2
                                ] = tmat[i_lam_1, idx_1, i_lam_2, idx_2]
                                idx_2 += 1
                        idx_1 += 1
    if num_k > 1:
        tmat_unwrapped = np.zeros(
            (num_k, 2, max_J, 2 * max_J + 1, 2, max_J, 2 * max_J + 1), dtype=complex
        )
        for i_k in range(num_k):
            for i_lam_1 in [0, 1]:
                for i_lam_2 in [0, 1]:
                    idx_1 = 0
                    for J_1 in range(1, max_J + 1):
                        i_J_1 = J_1 - 1
                        for m_1 in range(-J_1, J_1 + 1):
                            i_m_1 = m_1 + max_J
                            idx_2 = 0
                            for J_2 in range(1, max_J + 1):
                                i_J_2 = J_2 - 1
                                for m_2 in range(-J_2, J_2 + 1):
                                    i_m_2 = m_2 + max_J
                                    tmat_unwrapped[
                                        i_k,
                                        i_lam_1,
                                        i_J_1,
                                        i_m_1,
                                        i_lam_2,
                                        i_J_2,
                                        i_m_2,
                                    ] = tmat[i_k, i_lam_1, idx_1, i_lam_2, idx_2]
                                    idx_2 += 1
                            idx_1 += 1
    return tmat_unwrapped


def wrap_tmat(tmat_unwrapped, num_k, max_J):
    tmat = np.zeros(
        (num_k, 2, (max_J + 1) ** 2 - 1, 2, (max_J + 1) ** 2 - 1), dtype=complex
    )
    for i_k in range(num_k):
        for i_lam_1 in [0, 1]:
            for i_lam_2 in [0, 1]:
                idx_1 = 0
                for J_1 in range(1, max_J + 1):
                    i_J_1 = J_1 - 1
                    for m_1 in range(-J_1, J_1 + 1):
                        i_m_1 = m_1 + max_J
                        idx_2 = 0
                        for J_2 in range(1, max_J + 1):
                            i_J_2 = J_2 - 1
                            for m_2 in range(-J_2, J_2 + 1):
                                i_m_2 = m_2 + max_J
                                tmat[
                                    i_k, i_lam_1, idx_1, i_lam_2, idx_2
                                ] = tmat_unwrapped[
                                    i_k, i_lam_1, i_J_1, i_m_1, i_lam_2, i_J_2, i_m_2
                                ]
                                idx_2 += 1
                        idx_1 += 1

    return tmat


def unwrap_jm(idx_vals, num_k, max_J):
    if num_k > 1:
        jm_vals = np.zeros((2, num_k, max_J, 2 * max_J + 1), dtype=complex)
        for i_lam in [0, 1]:
            for i_k in range(num_k):
                idx = 0
                for J in range(1, max_J + 1):
                    i_J = J - 1
                    for m in range(-J, J + 1):
                        i_m = m + max_J
                        jm_vals[i_lam, i_k, i_J, i_m] = idx_vals[i_lam, i_k, idx]
                        idx += 1
    if num_k == 1:
        jm_vals = np.zeros((2, max_J, 2 * max_J + 1), dtype=complex)
        for i_lam in [0, 1]:
            idx = 0
            for J in range(1, max_J + 1):
                i_J = J - 1
                for m in range(-J, J + 1):
                    i_m = m + max_J
                    jm_vals[i_lam, i_J, i_m] = idx_vals[i_lam, idx]
                    idx += 1

    return jm_vals


def wrap_jm(jm_vals, num_k, max_J):
    if num_k > 1:
        idx_vals = np.zeros((2, num_k, (max_J + 1) ** 2 - 1), dtype=complex)
        for i_lam in [0, 1]:
            for i_k in range(num_k):
                idx = 0
                for J in range(1, max_J + 1):
                    i_J = J - 1
                    for m in range(-J, J + 1):
                        i_m = m + max_J
                        idx_vals[i_lam, i_k, idx] = jm_vals[i_lam, i_k, i_J, i_m]
                        idx += 1
    if num_k == 1:
        idx_vals = np.zeros((2, (max_J + 1) ** 2 - 1), dtype=complex)
        for i_lam in [0, 1]:
            idx = 0
            for J in range(1, max_J + 1):
                i_J = J - 1
                for m in range(-J, J + 1):
                    i_m = m + max_J
                    idx_vals[i_lam, idx] = jm_vals[i_lam, i_J, i_m]
                    idx += 1
    return idx_vals


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def scatter_representation_fancy(tmat, radius, incident):
    if not incident.info.parameters["radial_functions"] == ["j"]:
        raise TypeError("Incident field must be regular")
    else:
        scattered = copy.deepcopy(incident)
        if incident.info.parameters["num_k"] > 1:
            scattered.vals = [
                np.einsum(
                    "kajmbgn, kbgn->kajm",
                    tmat,
                    incident.vals[0],
                    optimize="greedy",
                )
            ]
            scattered.info.parameters["radial_functions"] = ["h+"]
            scattered.info.parameters["regions"] = [[radius, np.infty]]
        if incident.info.parameters["num_k"] == 1:
            scattered.vals = [
                np.einsum(
                    "ajmbgn, bgn->ajm",
                    tmat,
                    incident.vals[0],
                    optimize="greedy",
                )
            ]
            scattered.info.parameters["radial_functions"] = ["h+"]
            scattered.info.parameters["regions"] = [[radius, np.infty]]
        return scattered

def gaussian_wavenumber(k, k_0, dt):
    return np.exp(-(k - k_0) ** 2 * (dt * C_0) ** 2 / 2)


def get_k_list_gaussian(center_wavelength, width_time, num_k):
    """Gaussian width in time is implied"""
    k_0 = 2 * np.pi / center_wavelength
    width_k = 1 / (width_time * C_0)
    min_k = k_0 - 2.7 * width_k * np.sqrt(2)
    max_k = k_0 + 2.7 * width_k * np.sqrt(2)
    k_list = np.linspace(min_k, max_k, num_k, endpoint=True)
    if min_k <= 0:
        raise ValueError(f"Negative k in k_list: min_k = {min_k} ")
    return k_list


def get_k_list_narrow_gaussian(center_wavelength, width_time, num_k):
    """Gaussian width in time is implied"""
    k_0 = 2 * np.pi / center_wavelength
    width_k = 1 / (width_time * C_0)
    min_k = k_0 - 3.7 * width_k * np.sqrt(2)
    max_k = k_0 + 3.7 * width_k * np.sqrt(2)
    k_list = np.linspace(min_k, max_k, num_k, endpoint=True)
    if min_k <= 0:
        raise ValueError(f"Negative k in k_list: min_k = {min_k} ")
    return k_list


def get_m(idx, N_J):
    idx_dum = 0
    for J in range(1, N_J + 1):
        for m in range(-J, J + 1):
            if idx_dum == idx:
                return m
            idx_dum += 1


def get_J(idx, N_J):
    idx_dum = 0
    for J in range(1, N_J + 1):
        for m in range(-J, J + 1):
            if idx_dum == idx:
                return J
            idx_dum += 1


def get_idx(J_ref, m_ref):
    idx = 0
    for J in range(1, J_ref + 1):
        for m in range(-J, J + 1):
            if J_ref == J and m_ref == m:
                return idx
            idx += 1


def doubleTmatrix(Tmat_name):  # multiplies T-matrix by 2 to bring it to my convention
    with h5py.File(Tmat_name, "r+") as T_file:
        ifDoubled = T_file.require_dataset("ifDoubled", (1,), dtype=bool, exact=True)
        if T_file["ifDoubled"][0] == False:
            print("Multipying Tmat by two")
            T_mat = T_file["tmatrix"]
            T_mat[...] = np.array(T_mat) * 2
            T_file["ifDoubled"][0] = True


def get_full_silicon_keps(silicon_n_k_path_name):
    with open(silicon_n_k_path_name, "r") as f:
        lines = f.readlines()
        lines = lines[1:]
    lamnk_data_arr = np.zeros((np.shape(lines)[0], 3), dtype=float)
    for i_line, line in enumerate(lines):
        for i in range(3):
            lamnk_data_arr[i_line, i] = float(line.split()[i])
    wavelength_data_list = lamnk_data_arr[:, 0]
    k_data_list = 2 * np.pi / np.flip(wavelength_data_list)
    refr_data_list = np.flip(lamnk_data_arr[:, 1])
    ext_data_list = np.flip(lamnk_data_arr[:, 2])

    eps_list = (
        refr_data_list**2
        - ext_data_list**2
        + 1j * (refr_data_list * ext_data_list)
    )
    k_eps_arr = np.stack((k_data_list, eps_list), 0)
    return k_data_list, eps_list


def interpolate_silicon_epsilon(k_list, silicon_n_k_path_name):
    with open(silicon_n_k_path_name, "r") as f:
        lines = f.readlines()
        lines = lines[1:]
    lamnk_data_arr = np.zeros((np.shape(lines)[0], 3), dtype=float)
    for i_line, line in enumerate(lines):
        for i in range(3):
            lamnk_data_arr[i_line, i] = float(line.split()[i])
    wavelength_data_list = lamnk_data_arr[:, 0]
    k_data_list = 2 * np.pi / np.flip(wavelength_data_list)
    if k_list[0]<k_data_list[0] or k_list[-1]>k_data_list[-1]:
        raise ValueError(f'k_list outside of experimental values {k_list[0]}<{k_data_list[0]} or {k_list[-1]}>{k_data_list[-1]}')
    refr_data_list = np.flip(lamnk_data_arr[:, 1])
    ext_data_list = np.flip(lamnk_data_arr[:, 2])
    refr_interpol_list = np.interp(k_list, k_data_list, refr_data_list)
    ext_interpol_list = np.interp(k_list, k_data_list, ext_data_list)
    eps_list = (
        refr_interpol_list**2
        - ext_interpol_list**2
        + 1j * (refr_interpol_list * ext_interpol_list)
    )
    return eps_list





def getWF_scat(T_file, Wfunc_AM, k_list, N_J):  # Do in parametrization-independent way
    TmatBig = T_file["tmatrix"]
    N_k = np.shape(k_list)[0]
    Wfunc_scat_AM = np.zeros(
        (2, N_k, N_J * (N_J + 2)), complex
    )  # first dim is helicity starting positive
    Js = np.array(T_file["modes"]["l"])
    Lams = np.array(T_file["modes"]["polarization"])
    for i_k, k in np.ndenumerate(k_list):
        i_k = i_k[0]
        Tmat = np.squeeze(TmatBig[i_k, :, :])
        for Lam1 in [b"positive", b"negative"]:
            i_Lam1 = 0 if Lam1 == b"positive" else 1
            for Lam2 in [b"positive", b"negative"]:
                i_Lam2 = 0 if Lam2 == b"positive" else 1

                T_lam1_lam2 = Tmat[np.outer(Lams == Lam1, Lams == Lam2)]
                T_lam1_lam2 = T_lam1_lam2.reshape(N_J * (N_J + 2), N_J * (N_J + 2))

                Wfunc_scat_AM[i_Lam1, i_k, :] += np.dot(
                    T_lam1_lam2, Wfunc_AM[i_Lam2, i_k, :]
                )  # NO extra k is needed for diagonal in frequency Tmat

    return Wfunc_scat_AM


def scalar_AM(wfunc_1, wfunc_2, k_list):  # Can use non-uniform k_list discretization
    scal = 0
    for idx, val_1 in np.ndenumerate(wfunc_1):
        if idx[1] < np.shape(k_list)[0] - 1:
            delta_k = k_list[idx[1] + 1] - k_list[idx[1]]
            scal += delta_k * k_list[idx[1]] * np.conj(val_1) * wfunc_2[idx]
    return scal


def scalar_AM_per_k(wfunc_1, wfunc_2):  # WITHOUT k-multiplication
    scal = 0
    for idx, val_1 in np.ndenumerate(wfunc_1):
        scal += np.conj(val_1) * wfunc_2[idx]
    return scal


def checkCompatibilityTmatAndState(k_list, N_J, f):
    TmatBig = f["tmatrix"]

    if 2 * ((N_J + 1) ** 2 - 1) != np.shape(TmatBig)[1]:
        raise ValueError("No compatibility in N_J between Tmat and the state")

    # Scattered field can be computed only with this modes configuration
    Js = np.array(f["modes"]["l"])
    Ms = np.array(f["modes"]["m"])
    Lams = np.array(f["modes"]["polarization"])
    if np.array_equal(Ms[Lams == b"positive"][0:3], [-1, 0, 1]) == False:
        raise ValueError("Polarization struction of this T-matrix is not supported")
    if np.array_equal(Js[Lams == b"positive"][0:3], [1, 1, 1]) == False:
        raise ValueError("Polarization struction of this T-matrix is not supported")

    # print(Ms)
    # if not(f['modes']['polarization'][0]==b'positive' and f['modes']['polarization'][1]==b'negative'):
    #     raise ValueError('Polarization struction of this T-matrix is not supported')

    if np.shape(k_list)[0] != np.shape(TmatBig)[0]:
        raise ValueError(
            f"N_k={np.shape(k_list)[0]} not equal to amount of freqs in Tmatrix {np.shape(TmatBig)[0]}"
        )
    if not (
        isclose(f["angular_vacuum_wavenumber"][0], k_list[0])
        and isclose(f["angular_vacuum_wavenumber"][-1], k_list[-1])
    ):
        raise ValueError("Wavenumbers not corresponding to the pulse")

    print("T-mat and pulse are compatible\n")


def complex_quad(func, a, b, **kwargs):
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


def complex_dblquad(func, x0, x1, func0, func1, **kwargs):
    def real_func(y, x, *args):
        return np.real(func(y, x, *args))

    def imag_func(y, x, *args):
        return np.imag(func(y, x, *args))

    real_integral = dblquad(real_func, x0, x1, func0, func1, **kwargs)
    imag_integral = dblquad(imag_func, x0, x1, func0, func1, **kwargs)
    return (
        real_integral[0] + 1j * imag_integral[0],
        real_integral[1:],
        imag_integral[1:],
    )


# def get_polarizvec_complex_theta(lam, phi, theta, arg):
#     if lam == 1 or lam == -1:
#         Qx = lam * np.cos(phi) * np.cos(theta) - 1j * np.sin(phi)
#         Qy = lam * np.sin(phi) * np.cos(theta) + 1j * np.cos(phi)
#         Qz = -lam * np.sin(theta)
#         if arg == "vec":
#             return -1 / np.sqrt(2) * np.array([Qx, Qy, Qz])
#         if arg == "x":
#             return -1 / np.sqrt(2) * Qx
#         if arg == "y":
#             return -1 / np.sqrt(2) * Qy
#         if arg == "z":
#             return -1 / np.sqrt(2) * Qz
#     if lam == 0:
#         if arg == "vec":
#             return np.array(
#                 [
#                     np.cos(phi) * np.sin(theta),
#                     np.sin(phi) * np.sin(theta),
#                     np.cos(theta),
#                 ]
#             )
#         if arg == "x":
#             return np.cos(phi) * np.sin(theta)
#         if arg == "y":
#             return np.sin(phi) * np.sin(theta)
#         if arg == "z":
#             return np.cos(theta)


def get_polarizvec_complex(lam, phi, theta, arg):
    if lam == 1 or lam == -1:
        Qx = lam * np.cos(phi) * np.cos(theta) - 1j * np.sin(phi)
        Qy = lam * np.sin(phi) * np.cos(theta) + 1j * np.cos(phi)
        Qz = -lam * np.sin(theta)
        if arg == "vec":
            return -1 / np.sqrt(2) * np.array([Qx, Qy, Qz])
        if arg == "x":
            return -1 / np.sqrt(2) * Qx
        if arg == "y":
            return -1 / np.sqrt(2) * Qy
        if arg == "z":
            return -1 / np.sqrt(2) * Qz
    if lam == 0:
        if arg == "vec":
            return np.array(
                [
                    np.cos(phi) * np.sin(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(theta),
                ]
            )
        if arg == "x":
            return np.cos(phi) * np.sin(theta)
        if arg == "y":
            return np.sin(phi) * np.sin(theta)
        if arg == "z":
            return np.cos(theta)


def get_polarizvec_z(lam):
    if lam == -1 or lam == 1:
        return -1 / np.sqrt(2) * np.array([lam, 1j, 0], dtype=complex)
    if lam == 0:
        return np.array([0, 0, 1], dtype=complex)


def getY(J, m, theta_list, phi_list):  # first axis 3D vec, second axis L = J-1, J, J+1
    N_theta = len(theta_list)
    N_phi = len(phi_list)
    Y = np.zeros((3, 3, N_theta, N_phi), complex)
    for i_phi, phi in np.ndenumerate(phi_list):
        for i_theta, theta in np.ndenumerate(theta_list):
            for i_L in range(3):
                L = J + (i_L - 1)
                for i_sigma in range(3):
                    sigma = i_sigma - 1
                    if abs(m - sigma) <= L:
                        cg = clebsch_gordan(
                            L, m - sigma, 1, sigma, J, m
                        )  # (j1, m1, j2, m2, j3, m3)
                        wig = sp.wignerd(L, m - sigma, 0, phi, theta, 0)
                        Y[:, i_L, i_theta[0], i_phi[0]] = Y[
                            :, i_L, i_theta[0], i_phi[0]
                        ] + get_polarizvec_z(sigma) * np.conjugate(wig) * cg * np.sqrt(
                            2 * L + 1
                        )
                        # if i_L == 0:
                        # print("ConjWig:", np.conjugate(wig))
    return Y


def getV(J, m, lam, theta_list, phi_list):
    N_theta = len(theta_list)
    N_phi = len(phi_list)
    Y = getY(J, m, theta_list, phi_list)
    V = np.zeros((3, 3, N_theta, N_phi), complex)  # 3Dvec, L
    for i_L in range(3):
        L = J + (i_L - 1)
        cg = clebsch_gordan(L, 0, 1, lam, J, lam)  # (j1, m1, j2, m2, j3, m3)
        V[:, i_L, :, :] = (
            Y[:, i_L, :, :] * cg * np.sqrt((2 * L + 1) / (2 * J + 1)) * (1j) ** L
        )
    return V


def getR(k, J, m, lam, rho_list, theta_list, phi_list, t_list, typ, **kwargs):
    "Has k-factor"
    N_theta = len(theta_list)
    N_phi = len(phi_list)
    N_rho = len(rho_list)
    N_t = len(t_list)
    V = getV(J, m, lam, theta_list, phi_list)
    R = np.zeros((3, N_rho, N_theta, N_phi, N_t), complex)  # 3D
    sum_L = np.zeros((3, N_rho, N_theta, N_phi), complex)
    for i_L in range(3):
        L = J + (i_L - 1)
        for i_rho, rho in np.ndenumerate(rho_list):
            xk = rho * k
            if rho > kwargs.get("lower_limit", 0) and rho < kwargs.get(
                "higher_limit", float("inf")
            ):
                if typ == "j":  # and rho > rad:
                    sum_L[:, i_rho[0], :, :] = (
                        sum_L[:, i_rho[0], :, :] + spherical_jn(L, xk) * V[:, i_L, :, :]
                    )  # (spherical_jn(L, xk)+1j*spherical_yn(L, xk))
                if typ == "h+":  # and rho > kwargs.get('lower_limit',0):
                    sum_L[:, i_rho[0], :, :] = (
                        sum_L[:, i_rho[0], :, :]
                        + 0.5
                        * (spherical_jn(L, xk) + 1j * spherical_yn(L, xk))
                        * V[:, i_L, :, :]
                    )  # ! New convention with 1/2
                if typ == "h-":  # and rho > kwargs.get('lower_limit',0):
                    sum_L[:, i_rho[0], :, :] = (
                        sum_L[:, i_rho[0], :, :]
                        + 0.5
                        * (spherical_jn(L, xk) - 1j * spherical_yn(L, xk))
                        * V[:, i_L, :, :]
                    )  # ! New convention with 1/2
                if typ == "n":  # and rho > kwargs.get('lower_limit',0):
                    sum_L[:, i_rho[0], :, :] = (
                        sum_L[:, i_rho[0], :, :]
                        + (1j * spherical_yn(L, xk)) * V[:, i_L, :, :]
                    )
    for i_t, t in np.ndenumerate(t_list):
        R[:, :, :, :, i_t[0]] = (
            np.sqrt(4 * np.pi) * sum_L * k * np.exp(-1j * k * C0 * t)
        )  #### 4pi from integration and 1/sqrt(4pi) deep inside for spherical harmonics
    return R


def Factor_Polariz_Vec():  # No wavenumber k here
    Hbar = 1.054571817e-34
    C0_SI = 3e8
    Epsilon0 = 8.8541878128e-12
    return np.sqrt(C0_SI * Hbar / Epsilon0) / np.sqrt(2 * (2 * np.pi) ** 3)


def Factor_E_Field():  # Use to produce E-field in SI from f(k) in 1/nm
    return Factor_Polariz_Vec() * 1e-18  # account for nm^2


C0 = 300  # nm/fs
# Hbar = 1.054571817e-34
