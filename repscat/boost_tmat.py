"""
@author: Maxim Vavilin maxim@vavilin.de
"""

import numpy as np
import repscat as rs
import copy
import treams.special as sp
import matplotlib.pyplot as plt
import time


def get_k1_k2_lists(k_list, xi, len_k1, len_k2):
    k1_list = np.linspace(
        k_list[0] * np.exp(-np.abs(xi)), k_list[-1] * np.exp(np.abs(xi)), len_k1
    )
    k2_list = np.linspace(
        k_list[0] * np.exp(np.abs(xi)), k_list[-1] * np.exp(-np.abs(xi)), len_k2
    )
    return k1_list, k2_list

def get_k3_list(k1, k2, xi):
    k3_min = np.maximum(k1 * np.exp(-np.abs(xi)), k2 * np.exp(-np.abs(xi)))
    k3_max = np.minimum(k1 * np.exp(np.abs(xi)), k2 * np.exp(np.abs(xi)))
    num_k3 = np.maximum(200, int((k3_max - k3_min) * 500 / 0.02)) # density of information for silicon Tmat
    # num_k3 = 200
    return np.linspace(k3_min, k3_max, num_k3)


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


def boost_tmat_vals(xi, k_list, tmat_diag_vals, jay_max, len_k1, len_k2):
    """Get T(k1,k2) from T(k)
    Not linearized in k1 because interpolated matrix would be too huge.
    """
    k1_list, k2_list = get_k1_k2_lists(k_list, xi, len_k1, len_k2)

    tmat_boosted = np.zeros(
        (
            len_k1,
            len_k2,
            2,
            2,
            jay_max,
            jay_max,
            2 * jay_max + 1,
            2 * jay_max + 1,
        ),
        dtype=complex,
    )

    for i_k1, k1 in np.ndenumerate(k1_list):
        for i_k2, k2 in np.ndenumerate(k2_list):
            if k1 * np.exp(-2 * np.abs(xi)) < k2 < k1 * np.exp(2 * np.abs(xi)):
                k3_list = get_k3_list(k1, k2, xi)
                

                tmat_interpol = interpolate_tmat(k3_list, k_list, tmat_diag_vals)

                ct1 = (k1 * np.cosh(xi) - k3_list) / (k1 * np.sinh(xi))
                ct1p = (k1 - k3_list * np.cosh(xi)) / (k3_list * np.sinh(xi))
                ct2 = -(k3_list - k2 * np.cosh(xi)) / (k2 * np.sinh(xi))
                ct2p = -(k3_list * np.cosh(xi) - k2) / (k3_list * np.sinh(xi))

                if (
                    max(ct1) > 1.05
                    or max(ct1p) > 1.05
                    or max(ct2) > 1.05
                    or max(ct2p) > 1.05
                ):
                    raise ValueError("Bad cosine")

                k3_mask = (
                    (k2 * np.exp(-np.abs(xi)) < k3_list)
                    & (k3_list < k2 * np.exp(np.abs(xi)))
                ).astype(int)

                ct1 = np.clip(ct1, a_min=-1, a_max=1)
                ct1p = np.clip(ct1p, a_min=-1, a_max=1)
                ct2 = np.clip(ct2, a_min=-1, a_max=1)
                ct2p = np.clip(ct2p, a_min=-1, a_max=1)

                wig1 = np.zeros((jay_max, 2 * jay_max + 1, 2, len(k3_list)))
                wig1p = np.zeros((jay_max, 2 * jay_max + 1, 2, len(k3_list)))
                wig2 = np.zeros((jay_max, 2 * jay_max + 1, 2, len(k3_list)))
                wig2p = np.zeros((jay_max, 2 * jay_max + 1, 2, len(k3_list)))

                for jay in range(1, jay_max + 1):
                    fac_jay = np.sqrt(2 * jay + 1)
                    i_jay = jay - 1
                    for m in range(-jay, jay + 1):
                        i_m = m + jay_max
                        for i_lam, lam in np.ndenumerate(np.array([1, -1])):
                            wig1[i_jay, i_m, i_lam] = fac_jay * sp.wignersmalld(
                                jay, m, lam, np.arccos(ct1)
                            )
                            wig1p[i_jay, i_m, i_lam] = fac_jay * sp.wignersmalld(
                                jay, m, lam, np.arccos(ct1p)
                            )
                            wig2[i_jay, i_m, i_lam] = fac_jay * sp.wignersmalld(
                                jay, m, lam, np.arccos(ct2)
                            )
                            wig2p[i_jay, i_m, i_lam] = fac_jay * sp.wignersmalld(
                                jay, m, lam, np.arccos(ct2p)
                            )

                measure = (
                    np.diff(k3_list, append=np.squeeze(k3_list[-1])) / k3_list * k3_mask
                )

                # Debug
                if i_k1[0]==24 and i_k2[0]==22:
                    print('boost_vals:')
                    print('k2: ', k2, 'k_1 list: ', k1_list[0], k1_list[-1], len(k1_list))
                    print('k3_list: ', k3_list[0], k3_list[-1], len(k3_list))
                #     print(
                #         # 'k3_list[0],[1] ', k3_list[0], k3_list[1],
                #         # 'k3_list[1]-[0]', k3_list[1]-k3_list[0],
                #         # 'mink3, maxk3, lenk3 ', k3_list[0], k3_list[-1], len(k3_list),
                #         # 'Measure[0],[-2] ', measure[0], measure[-2],
                #         # 'wig1', wig1[0,jay_max+1,0,0], wig1[0,jay_max+1,0,-1],
                #         # 'k1 k3[-1] cos1p ', k1, k3_list[-1], ct1p[-1],'\n',
                #         # 'theta[-1], wig[-1] ', np.arccos(ct1p[-1]), sp.wignersmalld(0, jay_max+1,0, np.arccos(ct1p[-1])),'\n',
                #         # 'wig1p', wig1p[0,jay_max+1,0,0], wig1p[0,jay_max+1,0,-1],
                #         # 'k2 k3[0] cos2 ', k2, k3_list[0], ct2[0],'\n',
                #         'theta2[0], wig2[0] ', np.arccos(ct2[0]), sp.wignersmalld(1, 1, 1, np.arccos(ct2[0])),'\n',
                #         'wig2', wig2[0,jay_max+1,0,0], wig2[0,jay_max+1,0,-1],
                #         'wig2p', wig2p[0,jay_max+1,0,0], wig2p[0,jay_max+1,0,-1],
                #         'tmat interpol', tmat_interpol[0,0,0,0,0,jay_max+1, jay_max+1],  tmat_interpol[-1,0,0,0,0,jay_max+1, jay_max+1]
                #     )

                # Einsum legend: k=k3, t=t1,
                # i=j1, j=j2, r=j1prime, s=j2prime,
                # m=m1, n=m2, a=lam1, b=lam2
                tmat_boosted[i_k1, i_k2] = np.einsum(
                    "k,kabrsmn,imak,rmak,jnbk,snbk->abijmn",
                    measure,  # (k3)
                    tmat_interpol,  # (k3,lam,lam,j1prime,j2prime,m1,m2)
                    wig1,  # (jay1,m1,lam1,t1)
                    wig1p,  # (jay1prime,m1,lam1,t1)
                    wig2,  # (jay2prime,m2,lam2,t1,k2)
                    wig2p,  # (jay2,m2,lam2,t1,k2)
                ) / (
                    4 * k1 * k2 * np.sinh(xi) ** 2
                )  # one k3 is killed in the measure

    return tmat_boosted
