# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 11:27:25 2023

@author: Maxim Vavilin maxim@vavilin.de
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.patches as patches
import treams.special as sp
import pylab as pl

# import spherical
import treams.special as sp
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import time
from numpy import linalg as la
from matplotlib.ticker import MaxNLocator
from scipy.special import spherical_jn, spherical_yn
from wigners import wigner_3j, clebsch_gordan
from scipy.interpolate import interp1d


import repscat.aux_funcs as af
from repscat.constants import C_0, C_0_SI, H_BAR_SI, EPSILON_0_SI
import pickle


def get_field_from_wf_pw_polychrom(
    pw_rep, k_list, max_J, spacetime_domain, region, ket_type
):
    t_list = spacetime_domain["t_list"]
    x_list = spacetime_domain["x_list"]
    z_list = spacetime_domain["z_list"]
    k_list = pw_rep.domain["k_list"]
    eta_list = pw_rep.domain["eta_list"]
    eta_list = pw_rep.domain["phi_list"]

    # exp_k_t
    # exp_k_eta_z
    # polvec_lam_eta_phi
    exp_k_eta_phi_x = np.zeros(
        (len(k_list), len(eta_list), len(phi_list), len(x_list)), dtype=complex
    )
    for idx in np.ndindex(exp_k_eta_phi_x.shape()):
        print(idx)

    return 1
    # L_list = np.arange(0, max_J + 2)
    # m_list = np.arange(-max_J, max_J + 1)

    # trm_k = np.diff(k_list, append=k_list[-1]) * np.square(k_list)

    # exp_kt = np.zeros((len(k_list), len(t_list)), dtype=complex)
    # for i_k, k in np.ndenumerate(k_list):
    #     for i_t, t in np.ndenumerate(t_list):
    #         exp_kt[i_k, i_t] = np.exp(-1j * k * t * C_0)

    # bes = np.zeros((len(L_list), len(k_list), len(r_list)), dtype=complex)
    # for i_k, k in np.ndenumerate(k_list):
    #     for i_r, r in np.ndenumerate(r_list):
    #         if region[0] <= r < region[1]:
    #             if ket_type == "j":
    #                 bes[:, i_k, i_r] = spherical_jn(L_list, k * r).reshape(max_J + 2, 1)
    #             if ket_type == "h-" and r>0:
    #                 bes[:, i_k, i_r] = 0.5 * (
    #                     spherical_jn(L_list, k * r) - 1j * spherical_yn(L_list, k * r)
    #                 ).reshape(max_J + 2, 1)
    #             if ket_type == "h+" and r>0:
    #                 bes[:, i_k, i_r] = 0.5 * (
    #                     spherical_jn(L_list, k * r) + 1j * spherical_yn(L_list, k * r)
    #                 ).reshape(max_J + 2, 1)

    # cg_1_JL = np.zeros((max_J, 2, len(L_list)), dtype=complex)
    # for J in range(1, max_J + 1):
    #     i_J = J - 1
    #     trm_J = 1 / np.sqrt(2 * J + 1)
    #     for i_lam, lam in np.ndenumerate(np.array([1, -1])):
    #         for L in [J - 1, J, J + 1]:
    #             trm_L = (2 * L + 1) * (1j) ** L
    #             cg_1_JL[i_J, i_lam, L] = (
    #                 clebsch_gordan(L, 0, 1, lam, J, lam) * trm_J * trm_L
    #             )
    # polariz_is = np.zeros((3, 3), dtype=complex)
    # for i_s, s in np.ndenumerate(np.array([-1, 0, 1])):
    #     polariz_is[:, i_s] = af.get_polarizvec_z(s).reshape(3, 1)

    # small_wig = np.zeros((max_J + 2, 2 * max_J + 1, 3, len(theta_list)), dtype=float)
    # for L in L_list:
    #     for i_m, m in np.ndenumerate(np.arange(-max_J, max_J + 1)):
    #         for i_s, s in np.ndenumerate(np.array([-1, 0, 1])):
    #             if abs(m - s) <= L:
    #                 small_wig[L, i_m, i_s, :] = sp.wignersmalld(L, m - s, 0, theta_list)

    # exp_msphi = np.zeros((len(m_list), 3, len(phi_list)), dtype=complex)
    # for i_m, m in np.ndenumerate(m_list):
    #     for i_s, s in np.ndenumerate(np.array([-1, 0, 1])):
    #         exp_msphi[i_m, i_s, :] = np.exp(1j * (m - s) * phi_list).reshape(
    #             1, len(phi_list)
    #         )

    # cg_2 = np.zeros((max_J, len(m_list), len(L_list), 3), dtype=complex)
    # for J in range(1, max_J + 1):
    #     i_J = J - 1
    #     for m in range(-J, J + 1):
    #         i_m = m + max_J
    #         for L in [J - 1, J, J + 1]:
    #             for i_s, s in np.ndenumerate(np.array([-1, 0, 1])):
    #                 if abs(m - s) <= L:
    #                     cg_2[i_J, i_m, L, i_s] = clebsch_gordan(
    #                         L, m - s, 1, s, J, m
    #                     )  # (j1, m1, j2, m2, j3, m3)
    # units_factor = np.sqrt(C_0_SI * H_BAR_SI / EPSILON_0_SI) * 10**18 / (2 * np.pi)

    # path = [
    #     "einsum_path",
    #     (0, 2),
    #     (3, 5),
    #     (2, 4),
    #     (4, 5),
    #     (0, 3),
    #     (0, 3),
    #     (0, 1),
    #     (0, 1),
    # ]
    # return (
    #     np.einsum(
    #         "k,kajm,kt,lkr,jal,is,lmsh,msp,jmls->irhpt",
    #         trm_k,
    #         rep_vals,
    #         exp_kt,
    #         bes,
    #         cg_1_JL,
    #         polariz_is,
    #         small_wig,
    #         exp_msphi,
    #         cg_2,
    #         optimize=path,
    #     )
    #     * units_factor
    # )


def align_yaxis(ax1, ax2):
    """Align zeros of the two axes, zooming them out by same ratio"""
    axes = (ax1, ax2)
    extrema = [ax.get_ylim() for ax in axes]
    tops = [extr[1] / (extr[1] - extr[0]) for extr in extrema]
    # Ensure that plots (intervals) are ordered bottom to top:
    if tops[0] > tops[1]:
        axes, extrema, tops = [list(reversed(l)) for l in (axes, extrema, tops)]

    # How much would the plot overflow if we kept current zoom levels?
    tot_span = tops[1] + 1 - tops[0]

    b_new_t = extrema[0][0] + tot_span * (extrema[0][1] - extrema[0][0])
    t_new_b = extrema[1][1] - tot_span * (extrema[1][1] - extrema[1][0])
    axes[0].set_ylim(extrema[0][0], b_new_t)
    axes[1].set_ylim(t_new_b, extrema[1][1])


def get_data(
    EM_mesh, typ_data, r_list, theta2_list, N_t, N_theta
):
    N_phi = np.shape(EM_mesh)[-2]
    N_theta2 = np.shape(theta2_list)[0]
    N_rho = np.shape(r_list)[0]
    Data = np.zeros((N_theta2, N_rho, N_t), complex)
    for i_theta2, theta2 in np.ndenumerate(theta2_list):
        if i_theta2[0] < N_theta - 1:
            if typ_data == "E-density":
                EM_mesh_almost = np.squeeze(
                    EM_mesh[0, :, :, i_theta2[0], 0, :]
                    + np.conj(EM_mesh[1, :, :, i_theta2[0], 0, :])
                )
                Data[i_theta2] = (
                    np.einsum(
                        "abc,abc->bc",
                        EM_mesh_almost,
                        np.conj(EM_mesh_almost),
                        optimize=True,
                    )
                    * EPSILON_0_SI / 2
                )
            if typ_data == "norm":
                EM_mesh_nolam = np.einsum("abcdef->bcdef", EM_mesh)
                Data[i_theta2] = la.norm(EM_mesh_nolam[:, :, i_theta2[0], 0, :], axis=0)
            if typ_data == "2real0":
                EM_mesh_nolam = np.einsum("abcdef->bcdef", EM_mesh)
                Data[i_theta2] = 2 * np.real(EM_mesh_nolam[0, :, i_theta2[0], 0, :])
        else:
            if typ_data == "E-density":
                iphi = int((N_phi-1)/2)
                itheta = N_theta2 - i_theta2[0] - 1
                EM_mesh_almost = np.squeeze(
                    EM_mesh[0, :, :, itheta, 0, :]
                    + np.conj(EM_mesh[1, :, :, itheta, 0, :])
                )
                Data[i_theta2] = (
                    np.einsum(
                        "abc,abc->bc",
                        EM_mesh_almost,
                        np.conj(EM_mesh_almost),
                        optimize=True,
                    )
                    * EPSILON_0_SI / 2
                )
            if typ_data == "norm":
                EM_mesh_nolam = np.einsum("abcdef->bcdef", EM_mesh)
                Data[i_theta2] = la.norm(
                    EM_mesh_nolam[:, :, N_theta2 - i_theta2[0] - 1, iphi, :], axis=0
                )
            if typ_data == "2real0":
                EM_mesh_nolam = np.einsum("abcdef->bcdef", EM_mesh)
                Data[i_theta2] = 2 * np.real(
                    EM_mesh_nolam[0, :, N_theta2 - i_theta2[0] - 1, iphi, :]
                )
    return Data


def save_animation_profile(
    r_list, theta2_list, t_list, EM_mesh, name, typ_data, N_t, N_theta
):  # Animates Electric field on the x-axis
    Data = getData(EM_mesh, typ_data, r_list, theta2_list, N_t, N_theta)
    dpi_val = 220
    rho_mesh, theta2_mesh = np.meshgrid(r_list, theta2_list)
    fig, axs = plt.subplots(
        nrows=1, ncols=1, dpi=dpi_val, subplot_kw=dict(projection="polar"), sharex=True
    )
    maxval = np.max(np.abs(Data))
    minval = -maxval
    cmap = mpl.cm.seismic
    norm = mpl.colors.Normalize(vmin=minval, vmax=maxval)
    fig.subplots_adjust(
        bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.4, hspace=0.1
    )
    # fig.subplots_adjust(bottom=0.4, top=0.6, left=0.1, right=0.8,
    #                     wspace=0.4, hspace=0.1)
    cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax, orientation="vertical"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

    def animate(i_time):
        for idx, ax in np.ndenumerate(axs):
            ax.clear()
            # ax.contourf(theta2_mesh, rho_mesh, Data[:,:,i_time], 200, cmap=cmap, norm=norm)
            ax.pcolormesh(
                theta2_mesh, rho_mesh, Data[:, :, i_time], cmap=cmap, norm=norm
            )
            ax.grid(False)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.text(
                0.05,
                0.95,
                f"t = {int(t_list[i_time])}fs",
                fontsize=14,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=props,
            )

    ani = FuncAnimation(fig, animate, frames=np.arange(np.size(t_list)), blit=False)
    writervid = animation.writers["ffmpeg"](fps=30)
    ani.save(name + ".mp4", writer=writervid, dpi=dpi_val)

    # UP COMMEMNTED


def plotSnapshotTable(
    r_list, theta2_list, t_list, EM_mesh_list, typ_data, N_t, N_theta
):
    dpi_val = 220
    fig, axs = plt.subplots(
        nrows=3,
        ncols=N_t,
        dpi=dpi_val,
        subplot_kw=dict(projection="polar"),
        sharex=True,
    )
    rho_mesh, theta2_mesh = np.meshgrid(r_list, theta2_list)
    for row in range(3):
        Data = getData(EM_mesh_list[row], typ_data, r_list, theta2_list, N_t, N_theta)

        maxval = np.max(np.abs(Data))
        minval = -maxval if typ_data == "2real0" else 0
        cmap = mpl.cm.viridis  # mpl.cm.viridis
        norm = mpl.colors.Normalize(vmin=minval, vmax=maxval)

        for col in range(N_t):
            # axs[row,col].clear()
            axs[row, col].contourf(
                theta2_mesh,
                rho_mesh,
                np.squeeze(Data[:, :, col]),
                400,
                cmap=cmap,
                norm=norm,
            )
            # ax.pcolormesh(theta2_mesh, rho_mesh, np.squeeze(Data[:,:,i_ax]), cmap=cmap, norm=norm)
            axs[row, col].grid(False)
            axs[row, col].set_xticklabels([])
            axs[row, col].set_yticklabels([])
            if row == 0:
                axs[row, col].set_title(rf"$t = {int(t_list[col])}$fs")
            # if col == 0:
            # axs[row,col].set_ylabel('amplitude')
            # axs[row,col].text(-0.1, 0.95, f"t = {t_list[col]*1e3}ps", fontsize=14, transform=axs[row,col].transAxes,
            # verticalalignment='top', bbox=props)
    axs[0, 0].set_ylabel(r"$ \lvert \boldsymbol{E}_{p}(\boldsymbol{r}, t) \rvert$")
    axs[1, 0].set_ylabel(
        r"$ \lvert \boldsymbol{E}^{in}_{p}(\boldsymbol{r}, t) \rvert $"
    )
    axs[2, 0].set_ylabel(
        r"$ \lvert \boldsymbol{E}^{out}_{p}(\boldsymbol{r}, t) \rvert $"
    )

    fig.subplots_adjust(
        bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.4, hspace=0.1
    )
    cb_ax = fig.add_axes([0.87, 0.15, 0.02, 0.70])
    # cb_ax = fig.add_axes([0.87, 0.29, 0.02, 0.44])
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax, orientation="vertical"
    )
    cb_ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()

    # fig.suptitle('Propagation of three types of pulses', fontsize=16)


def get_field_from_wf_am(rep_wf_am, spacetime_domain, radial_function):
    r_list = spacetime_domain["r_list"]
    max_J = rep_wf_am.vals.shape[2]
    num_k = rep_wf_am.vals.shape[0]
    field = np.zeros(
        (
            3,
            len(r_list),
            len(spacetime_domain["theta_list"]),
            len(spacetime_domain["phi_list"]),
            len(spacetime_domain["t_list"]),
        ),
        dtype=complex,
    )

    region = spacetime_domain["region"]
    if num_k > 1:
        field = get_partial_field_from_wf_am_polychrom(
            rep_wf_am.vals, rep_wf_am.k_list, max_J, spacetime_domain, region, radial_function
        )

    if num_k == 1:
        raise NotImplementedError('Not implemented')

    return field

def get_field_of_k_from_wf_am(rep_wf_am, radtype, spacetime_domain):
    '''Electric field in (k, lam, r) in SI
    '''
    max_J = rep_wf_am.vals.shape[2] 
    ket_type = radtype
    region = spacetime_domain["region"]
    k_list = rep_wf_am.k_list
    r_list = spacetime_domain["r_list"]
    theta_list = spacetime_domain["theta_list"]
    phi_list = spacetime_domain["phi_list"]

    L_list = np.arange(0, max_J + 2)
    m_list = np.arange(-max_J, max_J + 1)

    trm_k = np.square(k_list)

    bes = np.zeros((len(L_list), len(k_list), len(r_list)), dtype=complex)
    for i_k, k in np.ndenumerate(k_list):
        for i_r, r in np.ndenumerate(r_list):
            if region[0] <= r < region[1]:
                if ket_type == "j":
                    bes[:, i_k, i_r] = spherical_jn(L_list, k * r).reshape(max_J + 2, 1)
                if ket_type == "h-" and r > 0:
                    bes[:, i_k, i_r] = 0.5 * (
                        spherical_jn(L_list, k * r) - 1j * spherical_yn(L_list, k * r)
                    ).reshape(max_J + 2, 1)
                if ket_type == "h+" and r > 0:
                    bes[:, i_k, i_r] = 0.5 * (
                        spherical_jn(L_list, k * r) + 1j * spherical_yn(L_list, k * r)
                    ).reshape(max_J + 2, 1)

    cg_1_JL = np.zeros((max_J, 2, len(L_list)), dtype=complex)
    for J in range(1, max_J + 1):
        i_J = J - 1
        trm_J = 1 / np.sqrt(2 * J + 1)
        for i_lam, lam in np.ndenumerate(np.array([1, -1])):
            for L in [J - 1, J, J + 1]:
                trm_L = (2 * L + 1) * (1j) ** L
                cg_1_JL[i_J, i_lam, L] = (
                    clebsch_gordan(L, 0, 1, lam, J, lam) * trm_J * trm_L
                )
    polariz_is = np.zeros((3, 3), dtype=complex)
    for i_s, s in np.ndenumerate(np.array([-1, 0, 1])):
        polariz_is[:, i_s] = af.get_polarizvec_z(s).reshape(3, 1)

    small_wig = np.zeros((max_J + 2, 2 * max_J + 1, 3, len(theta_list)), dtype=float)
    for L in L_list:
        for i_m, m in np.ndenumerate(np.arange(-max_J, max_J + 1)):
            for i_s, s in np.ndenumerate(np.array([-1, 0, 1])):
                if abs(m - s) <= L:
                    small_wig[L, i_m, i_s, :] = sp.wignersmalld(L, m - s, 0, theta_list)

    exp_msphi = np.zeros((len(m_list), 3, len(phi_list)), dtype=complex)
    for i_m, m in np.ndenumerate(m_list):
        for i_s, s in np.ndenumerate(np.array([-1, 0, 1])):
            exp_msphi[i_m, i_s, :] = np.exp(1j * (m - s) * phi_list).reshape(
                1, len(phi_list)
            )

    cg_2 = np.zeros((max_J, len(m_list), len(L_list), 3), dtype=complex)
    for J in range(1, max_J + 1):
        i_J = J - 1
        for m in range(-J, J + 1):
            i_m = m + max_J
            for L in [J - 1, J, J + 1]:
                for i_s, s in np.ndenumerate(np.array([-1, 0, 1])):
                    if abs(m - s) <= L:
                        cg_2[i_J, i_m, L, i_s] = clebsch_gordan(
                            L, m - s, 1, s, J, m
                        )  # (j1, m1, j2, m2, j3, m3)
    units_factor = np.sqrt(C_0_SI * H_BAR_SI / EPSILON_0_SI) * 10**9 / (2 * np.pi) * np.sqrt(2*np.pi)

    path = [
        "einsum_path",
        (3, 5),
        (2, 4),
        (4, 5),
        (0, 3),
        (0, 3),
        (0, 1),
        (0, 1),
    ]
    return (
        np.einsum(
            "kajm,lkr,jal,is,lmsh,msp,jmls,k->kairhp",
            rep_wf_am.vals,
            bes,
            cg_1_JL,
            polariz_is,
            small_wig,
            exp_msphi,
            cg_2,
            trm_k,
            optimize=path,
        )
        * units_factor
    )

def get_field_of_k_from_wf_am_on_array(rep_wf_am, rho_arr, theta_arr, phi_arr):
    '''Electric field in (k, lam, r) in SI
    '''
    bes_arr = spherical_jn(np.array([1,2,3,4,5]), rho_arr)
    print(np.shape(bes_arr))
    # max_J = rep_wf_am.info.parameters["max_J"]
    # ket_type = rep_wf_am.info.parameters["radial_function"]
    # region = rep_wf_am.info.parameters["region"]
    # k_list = rep_wf_am.domain["k_list"]
    # r_list = spacetime_domain["r_list"]
    # theta_list = spacetime_domain["theta_list"]
    # phi_list = spacetime_domain["phi_list"]

    # L_list = np.arange(0, max_J + 2)
    # m_list = np.arange(-max_J, max_J + 1)

    # trm_k = np.square(k_list)

    # bes = np.zeros((len(L_list), len(k_list), len(r_list)), dtype=complex)
    # for i_k, k in np.ndenumerate(k_list):
    #     for i_r, r in np.ndenumerate(r_list):
    #         if region[0] <= r < region[1]:
    #             if ket_type == "j":
    #                 bes[:, i_k, i_r] = spherical_jn(L_list, k * r).reshape(max_J + 2, 1)
    #             if ket_type == "h-" and r > 0:
    #                 bes[:, i_k, i_r] = 0.5 * (
    #                     spherical_jn(L_list, k * r) - 1j * spherical_yn(L_list, k * r)
    #                 ).reshape(max_J + 2, 1)
    #             if ket_type == "h+" and r > 0:
    #                 bes[:, i_k, i_r] = 0.5 * (
    #                     spherical_jn(L_list, k * r) + 1j * spherical_yn(L_list, k * r)
    #                 ).reshape(max_J + 2, 1)

    # cg_1_JL = np.zeros((max_J, 2, len(L_list)), dtype=complex)
    # for J in range(1, max_J + 1):
    #     i_J = J - 1
    #     trm_J = 1 / np.sqrt(2 * J + 1)
    #     for i_lam, lam in np.ndenumerate(np.array([1, -1])):
    #         for L in [J - 1, J, J + 1]:
    #             trm_L = (2 * L + 1) * (1j) ** L
    #             cg_1_JL[i_J, i_lam, L] = (
    #                 clebsch_gordan(L, 0, 1, lam, J, lam) * trm_J * trm_L
    #             )
    # polariz_is = np.zeros((3, 3), dtype=complex)
    # for i_s, s in np.ndenumerate(np.array([-1, 0, 1])):
    #     polariz_is[:, i_s] = af.get_polarizvec_z(s).reshape(3, 1)

    # small_wig = np.zeros((max_J + 2, 2 * max_J + 1, 3, len(theta_list)), dtype=float)
    # for L in L_list:
    #     for i_m, m in np.ndenumerate(np.arange(-max_J, max_J + 1)):
    #         for i_s, s in np.ndenumerate(np.array([-1, 0, 1])):
    #             if abs(m - s) <= L:
    #                 small_wig[L, i_m, i_s, :] = sp.wignersmalld(L, m - s, 0, theta_list)

    # exp_msphi = np.zeros((len(m_list), 3, len(phi_list)), dtype=complex)
    # for i_m, m in np.ndenumerate(m_list):
    #     for i_s, s in np.ndenumerate(np.array([-1, 0, 1])):
    #         exp_msphi[i_m, i_s, :] = np.exp(1j * (m - s) * phi_list).reshape(
    #             1, len(phi_list)
    #         )

    # cg_2 = np.zeros((max_J, len(m_list), len(L_list), 3), dtype=complex)
    # for J in range(1, max_J + 1):
    #     i_J = J - 1
    #     for m in range(-J, J + 1):
    #         i_m = m + max_J
    #         for L in [J - 1, J, J + 1]:
    #             for i_s, s in np.ndenumerate(np.array([-1, 0, 1])):
    #                 if abs(m - s) <= L:
    #                     cg_2[i_J, i_m, L, i_s] = clebsch_gordan(
    #                         L, m - s, 1, s, J, m
    #                     )  # (j1, m1, j2, m2, j3, m3)
    # units_factor = np.sqrt(C_0_SI * H_BAR_SI / EPSILON_0_SI) * 10**9 / (2 * np.pi) * np.sqrt(2*np.pi)

    # path = [
    #     "einsum_path",
    #     (3, 5),
    #     (2, 4),
    #     (4, 5),
    #     (0, 3),
    #     (0, 3),
    #     (0, 1),
    #     (0, 1),
    # ]
    # return (
    #     np.einsum(
    #         "kajm,lkr,jal,is,lmsh,msp,jmls,k->kairhp",
    #         rep_wf_am.vals,
    #         bes,
    #         cg_1_JL,
    #         polariz_is,
    #         small_wig,
    #         exp_msphi,
    #         cg_2,
    #         trm_k,
    #         optimize=path,
    #     )
    #     * units_factor
    # )

def get_spacetime_domain(plot_params):
    num_r = plot_params["num_r"]
    num_theta = plot_params["num_theta"]
    t_list = plot_params["t_list"]
    max_r = plot_params["max_r"]
    num_phi = plot_params["num_phi"]
    region = plot_params["region"]
    # if num_phi%2 == 0:
    #     raise ValueError('num_phi must be odd')

    max_theta = np.pi
    min_theta = 0
    theta_list = np.linspace(min_theta, max_theta, num_theta, endpoint=True)

    min_phi = 0
    max_phi = 2*np.pi
    phi_list = np.linspace(min_phi, max_phi, num_phi, endpoint=True)

    min_r = 0
    
    r_list = np.linspace(min_r, max_r, num_r, endpoint=True)

    spacetime_domain = {
        "r_list": r_list,
        "theta_list": theta_list,
        "phi_list": phi_list,
        "t_list": t_list,
        "region": region,
    }
    return spacetime_domain

def plot_field_from_wf_am(rep_wf_am, plot_params, radial_function, draw_patch=False):
    if plot_params['num_phi']%2 == 0:
            raise ValueError('num_phi must be odd')
    spacetime_domain = get_spacetime_domain(plot_params)
    
    t1 = time.time()
    field_vals = get_field_from_wf_am(rep_wf_am, spacetime_domain, radial_function)
    t2 = time.time()
    # print(f"Time for vals: {t2-t1}s")
    
    type_data = plot_params["what_to_plot"]

    num_theta2 = 2 * len(spacetime_domain["theta_list"]) - 1 # Must be -1
    theta2_list = np.linspace(0, 2 * np.pi, num_theta2, endpoint=True)

    r_list = spacetime_domain["r_list"]
    theta_list = spacetime_domain["theta_list"]
    t_list = spacetime_domain["t_list"]


    data=get_data(
        field_vals, type_data, r_list, theta2_list, len(t_list), len(theta_list)
    )
    maxval = np.max(np.abs(data))
    minval = -maxval if type_data == "2real0" else 0
    dpi_val = 220
    rho_mesh, theta2_mesh = np.meshgrid(r_list, theta2_list)
    fig, axs = plt.subplots(
        nrows=1,
        ncols=len(t_list),
        dpi=dpi_val,
        subplot_kw=dict(projection="polar"),
        sharex=True,
    )
    
    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=minval, vmax=maxval)
    fig.subplots_adjust(
        bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.4, hspace=0.1
    )
    
    cb_ax = fig.add_axes([0.87, 0.32, 0.02, 0.38])

    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax, orientation="vertical"
    )
    cb_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    cb_ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    for i_ax, ax in np.ndenumerate(axs):
        # ax.contourf(
        #     theta2_mesh,
        #     rho_mesh,
        #     np.squeeze(data[:, :, i_ax]),
        #     400,
        #     cmap=cmap,
        #     norm=norm,
        # )
        ax.pcolormesh(
            theta2_mesh, rho_mesh, np.real(np.squeeze(data[:, :, i_ax])), cmap=cmap, norm=norm
        )

        ### Add circle to plot
        circle = pl.Circle((0, 0), 200, transform=ax.transData._b, color='w')
        ax.add_artist(circle)

        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        if draw_patch == True:
            # Text with scale


            if plot_params['max_r']==50000:
                ax.text(1.17*np.pi, 1.3*25000, '$50~\mu$m', color='white',
                rotation=90, rotation_mode='anchor', fontsize=8)
            if plot_params['max_r']==5000:
                ax.text(1.1*np.pi, 1.15*2500, '$5~\mu$m', color='white',
                rotation=90, rotation_mode='anchor', fontsize=8)
            if plot_params['max_r']==500:
                ax.text(1.1*np.pi, 1.15*250, '$500~$nm', color='white',
                rotation=90, rotation_mode='anchor', fontsize=8)

            rad_patch = plot_params['max_r']/2
            for curve in [[[0, 2 * np.pi], [rad_patch, rad_patch]]]:
                x = np.linspace(curve[0][0], curve[0][1], 500)
                y = interp1d(curve[0], curve[1])(x)
                ax.plot(x, y, color="white", linewidth=0.75, alpha=1)
            curves = []
            for n in range(4):
                curves.append(
                    [
                        [np.pi / 4 + n * np.pi / 2, -np.pi / 4 + n * np.pi / 2],
                        [rad_patch*np.sqrt(2), rad_patch*np.sqrt(2)],
                    ]
                )
            for curve in curves:
                ax.plot(
                    curve[0],
                    curve[1],
                    color="white",
                    linestyle="--",
                    linewidth=0.75,
                    alpha=1,
                )

    cb_ax.set_ylabel(r'Energy density, J/m$^3$', fontsize=13)

    plt.savefig(f"figs/pulse_{plot_params['max_r']}.png", bbox_inches = "tight", dpi = 300)
    # plt.show()


def get_partial_field_from_wf_am_monochrom(
    rep_vals, k, max_J, spacetime_domain, region, ket_type
):
    t_list = spacetime_domain["t_list"]
    r_list = spacetime_domain["r_list"]
    theta_list = spacetime_domain["theta_list"]
    phi_list = spacetime_domain["phi_list"]

    L_list = np.arange(0, max_J + 2)
    m_list = np.arange(-max_J, max_J + 1)

    wf_am_vals_unwrapped = af.unwrap_jm(rep_vals, 1, max_J)

    exp_kt = np.zeros((len(t_list)), dtype=complex)
    for i_t, t in np.ndenumerate(t_list):
        exp_kt[i_t] = np.exp(-1j * k * t * C_0)

    bes = np.zeros((len(L_list), len(r_list)), dtype=complex)
    for i_r, r in np.ndenumerate(r_list):
        if region[0] <= r < region[1]:
            if ket_type == "j":
                bes[:, i_r] = spherical_jn(L_list, k * r).reshape(max_J + 2, 1)
            if ket_type == "h-" and r > 0:
                bes[:, i_r] = 0.5 * (
                    spherical_jn(L_list, k * r) - 1j * spherical_yn(L_list, k * r)
                ).reshape(max_J + 2, 1)
            if ket_type == "h+" and r > 0:
                bes[:, i_r] = 0.5 * (
                    spherical_jn(L_list, k * r) + 1j * spherical_yn(L_list, k * r)
                ).reshape(max_J + 2, 1)

    cg_1_JL = np.zeros((max_J, 2, len(L_list)), dtype=complex)
    for J in range(1, max_J + 1):
        i_J = J - 1
        trm_J = 1 / np.sqrt(2 * J + 1)
        for i_lam, lam in np.ndenumerate(np.array([1, -1])):
            for L in [J - 1, J, J + 1]:
                trm_L = (2 * L + 1) * (1j) ** L
                cg_1_JL[i_J, i_lam, L] = (
                    clebsch_gordan(L, 0, 1, lam, J, lam) * trm_J * trm_L
                )
    polariz_is = np.zeros((3, 3), dtype=complex)
    for i_s, s in np.ndenumerate(np.array([-1, 0, 1])):
        polariz_is[:, i_s] = af.get_polarizvec_z(s).reshape(3, 1)

    small_wig = np.zeros((max_J + 2, 2 * max_J + 1, 3, len(theta_list)), dtype=float)
    for L in L_list:
        for i_m, m in np.ndenumerate(np.arange(-max_J, max_J + 1)):
            for i_s, s in np.ndenumerate(np.array([-1, 0, 1])):
                if abs(m - s) <= L:
                    small_wig[L, i_m, i_s, :] = sp.wignersmalld(L, m - s, 0, theta_list)

    exp_msphi = np.zeros((len(m_list), 3, len(phi_list)), dtype=complex)
    for i_m, m in np.ndenumerate(m_list):
        for i_s, s in np.ndenumerate(np.array([-1, 0, 1])):
            exp_msphi[i_m, i_s, :] = np.exp(1j * (m - s) * phi_list).reshape(
                1, len(phi_list)
            )

    cg_2 = np.zeros((max_J, len(m_list), len(L_list), 3), dtype=complex)
    for J in range(1, max_J + 1):
        i_J = J - 1
        for m in range(-J, J + 1):
            i_m = m + max_J
            for L in [J - 1, J, J + 1]:
                for i_s, s in np.ndenumerate(np.array([-1, 0, 1])):
                    if abs(m - s) <= L:
                        cg_2[i_J, i_m, L, i_s] = clebsch_gordan(
                            L, m - s, 1, s, J, m
                        )  # (j1, m1, j2, m2, j3, m3)
    units_factor = np.sqrt(C_0_SI * H_BAR_SI / EPSILON_0_SI) * 10**18 / (2 * np.pi)

    # path = [
    #     "einsum_path",
    #     (0, 2),
    #     (3, 5),
    #     (2, 4),
    #     (4, 5),
    #     (0, 3),
    #     (0, 3),
    #     (0, 1),
    #     (0, 1),
    # ]
    return (
        np.einsum(
            "ajm,t,lr,jal,is,lmsh,msp,jmls->irhpt",
            wf_am_vals_unwrapped,
            exp_kt,
            bes,
            cg_1_JL,
            polariz_is,
            small_wig,
            exp_msphi,
            cg_2,
            optimize="greedy",
        )
        * units_factor
    )


def get_partial_field_from_wf_am_polychrom(
    rep_vals, k_list, max_J, spacetime_domain, region, ket_type
):
    t_list = spacetime_domain["t_list"]
    r_list = spacetime_domain["r_list"]
    theta_list = spacetime_domain["theta_list"]
    phi_list = spacetime_domain["phi_list"]

    L_list = np.arange(0, max_J + 2)
    m_list = np.arange(-max_J, max_J + 1)

    trm_k = np.diff(k_list, append=k_list[-1]) * np.square(k_list)

    exp_kt = np.zeros((len(k_list), len(t_list)), dtype=complex)
    for i_k, k in np.ndenumerate(k_list):
        for i_t, t in np.ndenumerate(t_list):
            exp_kt[i_k, i_t] = np.exp(-1j * k * t * C_0)

    bes = np.zeros((len(L_list), len(k_list), len(r_list)), dtype=complex)
    for i_k, k in np.ndenumerate(k_list):
        for i_r, r in np.ndenumerate(r_list):
            if region[0] <= r < region[1]:
                if ket_type == "j":
                    bes[:, i_k, i_r] = spherical_jn(L_list, k * r).reshape(max_J + 2, 1)
                if ket_type == "h-" and r > 0:
                    bes[:, i_k, i_r] = 0.5 * (
                        spherical_jn(L_list, k * r) - 1j * spherical_yn(L_list, k * r)
                    ).reshape(max_J + 2, 1)
                if ket_type == "h+" and r > 0:
                    bes[:, i_k, i_r] = 0.5 * (
                        spherical_jn(L_list, k * r) + 1j * spherical_yn(L_list, k * r)
                    ).reshape(max_J + 2, 1)

    cg_1_JL = np.zeros((max_J, 2, len(L_list)), dtype=complex)
    for J in range(1, max_J + 1):
        i_J = J - 1
        trm_J = 1 / np.sqrt(2 * J + 1)
        for i_lam, lam in np.ndenumerate(np.array([1, -1])):
            for L in [J - 1, J, J + 1]:
                trm_L = (2 * L + 1) * (1j) ** L
                cg_1_JL[i_J, i_lam, L] = (
                    clebsch_gordan(L, 0, 1, lam, J, lam) * trm_J * trm_L
                )
    polariz_is = np.zeros((3, 3), dtype=complex)
    for i_s, s in np.ndenumerate(np.array([-1, 0, 1])):
        polariz_is[:, i_s] = af.get_polarizvec_z(s).reshape(3, 1)

    small_wig = np.zeros((max_J + 2, 2 * max_J + 1, 3, len(theta_list)), dtype=float)
    for L in L_list:
        for i_m, m in np.ndenumerate(np.arange(-max_J, max_J + 1)):
            for i_s, s in np.ndenumerate(np.array([-1, 0, 1])):
                if abs(m - s) <= L:
                    small_wig[L, i_m, i_s, :] = sp.wignersmalld(L, m - s, 0, theta_list)

    exp_msphi = np.zeros((len(m_list), 3, len(phi_list)), dtype=complex)
    for i_m, m in np.ndenumerate(m_list):
        for i_s, s in np.ndenumerate(np.array([-1, 0, 1])):
            exp_msphi[i_m, i_s, :] = np.exp(1j * (m - s) * phi_list).reshape(
                1, len(phi_list)
            )

    cg_2 = np.zeros((max_J, len(m_list), len(L_list), 3), dtype=complex)
    for J in range(1, max_J + 1):
        i_J = J - 1
        for m in range(-J, J + 1):
            i_m = m + max_J
            for L in [J - 1, J, J + 1]:
                for i_s, s in np.ndenumerate(np.array([-1, 0, 1])):
                    if abs(m - s) <= L:
                        cg_2[i_J, i_m, L, i_s] = clebsch_gordan(
                            L, m - s, 1, s, J, m
                        )  # (j1, m1, j2, m2, j3, m3)
    units_factor = np.sqrt(C_0_SI * H_BAR_SI / EPSILON_0_SI) * 10**18 / (2 * np.pi)

    path = [
        "einsum_path",
        (0, 2),
        (3, 5),
        (2, 4),
        (4, 5),
        (0, 3),
        (0, 3),
        (0, 1),
        (0, 1),
    ]
    return (
        np.einsum(
            "k,kajm,kt,lkr,jal,is,lmsh,msp,jmls->airhpt",
            trm_k,
            rep_vals,
            exp_kt,
            bes,
            cg_1_JL,
            polariz_is,
            small_wig,
            exp_msphi,
            cg_2,
            optimize=path,
        )
        * units_factor
    )


def plot_irregular_wf_am(rep_wf_am, spacetime_domain):
    t1 = time.time()
    field_vals = get_field_from_wf_am(rep_wf_am, spacetime_domain)
    t2 = time.time()
    print(f"Time for vals: {t2-t1}s")
    type_data = "norm"  # '2real0'

    num_theta2 = 2 * len(spacetime_domain["theta_list"]) - 1  # Must be -1
    theta2_list = np.linspace(0, 2 * np.pi, num_theta2, endpoint=True)

    r_list = spacetime_domain["r_list"]
    theta_list = spacetime_domain["theta_list"]
    t_list = spacetime_domain["t_list"]

    data = get_data(
        field_vals, type_data, r_list, theta2_list, len(t_list), len(theta_list)
    )
    dpi_val = 220
    rho_mesh, theta2_mesh = np.meshgrid(r_list, theta2_list)
    fig, axs = plt.subplots(
        nrows=1,
        ncols=len(t_list),
        dpi=dpi_val,
        subplot_kw=dict(projection="polar"),
        sharex=True,
    )
    maxval = np.max(np.abs(data))
    minval = -maxval if type_data == "2real0" else 0
    cmap = mpl.cm.viridis  # mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=minval, vmax=maxval)
    fig.subplots_adjust(
        bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.4, hspace=0.1
    )
    # cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
    cb_ax = fig.add_axes([0.87, 0.29, 0.02, 0.44])
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax, orientation="vertical"
    )
    cb_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    props = dict(boxstyle="round", facecolor="white", alpha=1)

    for i_ax, ax in np.ndenumerate(axs):
        # ax.contourf(
        #     theta2_mesh,
        #     rho_mesh,
        #     np.squeeze(data[:, :, i_ax]),
        #     400,
        #     cmap=cmap,
        #     norm=norm,
        # )
        ax.pcolormesh(
            theta2_mesh, rho_mesh, np.squeeze(data[:, :, i_ax]), cmap=cmap, norm=norm
        )
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.text(
            0,
            2.5,
            f"t = {t_list[i_ax]}fs",
            fontsize=11,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=props,
        )
    plt.show()


def animateBasis(J, m, lam, typ, nameFile):  # typ: h+, h-, j
    # wigner = spherical.Wigner(J)
    # Physical constants
    # wlen_center = 780 # 800.0 # 400 nm # 350 also there
    # k0 = 2*np.pi/wlen_center # nm
    # dt = 10 # fs pulse width

    C0 = 300
    # Dt2C2 = (dt*C0)**2/4
    # dk = 1/np.sqrt(Dt2C2)
    k0 = 20
    dk = 6
    Min_k = k0 - 2.7 * dk  # 2.7
    if Min_k <= 0:
        raise ValueError(f"negative k_list!: Min_k = {Min_k} ")
    Max_k = k0 + 2.7 * dk

    # lambda0_min = 650.0 # nm
    # lambda0_max = 1010.0 # nm
    # lambda00 = 780.0 # nm
    # Max_k = 2*np.pi/lambda0_min
    # Min_k = 2*np.pi/lambda0_max

    N_k = 150  # 150 is enough

    k_list = np.linspace(Min_k, Max_k, N_k, endpoint=True)

    # fig, ax = plt.subplots()
    # ax.plot(k_list, np.exp(-(k_list-k0)**2/dk**2))

    # Spacetime Grid parameters
    N_theta = 10  # 50
    Max_theta = np.pi
    Min_theta = 0
    theta_list = np.linspace(Min_theta, Max_theta, N_theta, endpoint=True)

    N_J = 8
    wigner = spherical.Wigner(N_J + 1)

    # width_k = 1/np.sqrt(Dt2C2)

    N_theta2 = 2 * N_theta - 1
    theta2_list = np.linspace(0, 2 * np.pi, N_theta2, endpoint=True)

    N_rho = 100  # 70 is good
    Min_rho = 0
    Max_rho = 5  # 12000
    r_list = np.linspace(Min_rho, Max_rho, N_rho, endpoint=True)
    rad = 0.1 * Max_rho

    N_t = 250  # need 100
    Min_t = -3 / 300
    Max_t = 3 / 300
    t_list = np.linspace(Min_t, Max_t, N_t, endpoint=True)

    N_phi = 2  # 100 is good?
    Min_phi = 0
    Max_phi = np.pi
    phi_list = np.linspace(Min_phi, Max_phi, N_phi, endpoint=True)

    N_J = 8
    # wigner = spherical.Wigner(N_J+1)

    # Dt2C2 = (dt*C0)**2/4

    # N_k = np.shape(k_list)[0]
    delta_k = k_list[1] - k_list[0]

    print(f"Wavelength region:{int(2*np.pi/Max_k)}nm to {int(2*np.pi/Min_k)}nm")

    EM_mesh_basis_pulse = np.zeros(
        (3, N_rho, N_theta, N_phi, N_t), complex
    )  # R = np.zeros((3, N_rho, N_theta, N_phi, N_t), complex) #3D

    # with open(f'Wfunc_AM_Scat_{N_k}_{N_J}.npy', 'rb') as f:
    #     Wfunc_scat = np.load(f)

    # N_J_trunk = 4
    # EM_mesh_scat = np.zeros((3, N_rho, N_theta, N_phi, N_t), complex) #R = np.zeros((3, N_rho, N_theta, N_phi, N_t), complex) #3D
    # for i_k, k in np.ndenumerate(k_list):
    #     if (i_k[0])%(N_k//20+1) == 0:
    #         print(f"Momentum integration progress: {i_k[0]}/{N_k}")
    #     dkk = delta_k*k
    #     for lam in range(2): # MISTAKE!!!
    #         idx_jm = 0
    #         for J in range(1, N_J_trunk+1):
    #             for m in range(-J, J+1):
    #                 EM_mesh_scat += dkk*Wfunc_scat[lam, i_k, idx_jm]*getR(k, J, m, lam, r_list, theta_list, phi_list, t_list, wigner, typ, rad)
    #                 idx_jm += 1

    ## Figure out why h- does not go into h+

    # EM_mesh_basis_pulse = getR(k0, J, m, lam, r_list, theta_list, phi_list, t_list, wigner, typ, rad)
    ## GOOD PULSE !
    EM_mesh_basis_pulse = np.zeros(
        (3, N_rho, N_theta, N_phi, N_t), complex
    )  # R = np.zeros((3, N_rho, N_theta, N_phi, N_t), complex) #3D
    for i_k, k in np.ndenumerate(k_list):
        if (i_k[0]) % (N_k // 20 + 1) == 0:
            print(f"Momentum integration progress: {i_k[0]}/{N_k}")
        dkk = delta_k * k
        Rhplus = af.getR(
            k, J, m, lam, r_list, theta_list, phi_list, t_list, typ, rad
        )  # np.zeros((3, N_rho, N_theta, N_phi, N_t)
        # Rn = getR(k, J, m, lam, r_list, theta_list, phi_list, t_list, wigner, 'n', rad)
        # if i_k[0]==0:
        #     return
        # print(k, J, m, lam, r_list, theta_list, phi_list, t_list, wigner, typ, rad)
        # print('exp', np.exp(-(k-k0)**2*Dt2C2))
        # print('Rhplus',  Rhplus[0,15,15,0,:])
        EM_mesh_basis_pulse += dkk * Rhplus * np.exp(-((k - k0) ** 2) / dk**2)

    # print(EM_mesh_basis_pulse[0,15,15,0,:])
    ## QUARATURE PULSE vs what I get
    # print(EM_mesh_basis_pulse[0,25, 10, 0, :])
    # print(np.max(np.abs(EM_mesh_basis_pulse[0,25, 10, 0, :])))
    # print(np.max(np.abs(EM_mesh_basis_pulse[0,25, 10, 0, :3])))
    # print(np.max(np.abs(EM_mesh_basis_pulse[0,25, 10, 0, :10])))
    # def integrand_basis_pulse(k, axis):
    #     print(np.exp(-(k-k0)**2*Dt2C2)*getR(k, J, m, lam, r_list, theta_list, phi_list, t_list, wigner, typ, rad)[axis])
    #     return np.exp(-(k-k0)**2*Dt2C2)*getR(k, J, m, lam, r_list, theta_list, phi_list, t_list, wigner, typ, rad)[axis]

    # EM_mesh_basis_pulse = complex_quad(integrand_basis_pulse, Min_k, Max_k, args = (0))# args=(axis, L, t, rho, typ))

    # EM_mesh_scat = getR(k0, J, m, lam, r_list, theta_list, phi_list, t_list, wigner, typ, rad)

    # EM_mesh_scat = getR_integrated(J, m, lam, r_list, theta_list, phi_list, t_list, wigner, typ, rad)

    # def integrand_basis_pulse(p, axis):
    #     return np.exp(-(p-p0)**2*Dt2C2)*getR(p, J, m, lam, r_list, theta_list, phi_list, t_list, wigner, typ, rad)[axis]

    # saved_mesh = EM_mesh_scat

    # ANIMATE
    t2 = time.time()
    typ_data = "real0"  # "real0"
    name_anim = f"Anim_Scat5_{N_t}_{J}_{m}_{lam}_{typ_data}_{typ}"
    # name_anim = nameFile
    save_animation_profile(
        r_list,
        theta2_list,
        t_list,
        EM_mesh_basis_pulse,
        name_anim,
        typ_data,
        N_t,
        N_theta,
    )

    # save_animation_profile(r_list, theta2_list, t_list, saved_mesh , name_anim, typ_data, N_t, N_theta)

    # save_animation_profile(r_list, theta_list, t_list, getR(k, J, m, lam, r_list, theta_list, phi_list, t_list, wigner, typ, rad), name_anim, typ_data)
    #### save_animation_profile_cartes(r_list, theta_list, t_list, EM_mesh_scat, name_anim, typ_data)
    t3 = time.time()
    print(f"Finished animating in {t3-t2}s")
