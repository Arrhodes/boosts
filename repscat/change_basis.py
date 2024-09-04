"""
@author: Maxim Vavilin maxim@vavilin.de
"""

import numpy as np
import treams.special as sp


def pw_from_am_vals(vals_am, theta_list, phi_list):
    if len(vals_am.shape) != 4:
        raise ValueError("Monochromatic not implemented")
    max_jay = np.shape(vals_am)[2]

    m_phi_mat = np.zeros((2 * max_jay + 1, len(phi_list)), dtype=complex)
    for m in range(-max_jay, max_jay + 1):
        i_m = m + max_jay
        m_phi_mat[i_m, :] = np.exp(1j * m * phi_list)  # No need for np.diff here

    lam_j_m_theta_mat = np.zeros(
        (2, max_jay, 2 * max_jay + 1, len(theta_list)), dtype=complex
    )
    for i_lam in [0, 1]:
        lam = 1 if i_lam == 0 else -1
        for jay in range(1, max_jay + 1):
            i_jay = jay - 1
            fac_jay = np.sqrt((2 * jay + 1) / (4 * np.pi))
            for m in range(-jay, jay + 1):
                i_m = m + max_jay
                lam_j_m_theta_mat[i_lam, i_jay, i_m, :] = (
                    sp.wignersmalld(jay, m, lam, theta_list)
                    # No need for np.diff here
                    * fac_jay
                )
    vals_pw = np.einsum(
        "ajme,mp,kajm->kaep", lam_j_m_theta_mat, m_phi_mat, vals_am, optimize=True
    )
    return vals_pw



def am_from_pw_vals(vals_pw, theta_list, phi_list, max_jay):
    m_phi_mat = np.zeros((2 * max_jay + 1, len(phi_list)), dtype=complex)
    for m in range(-max_jay, max_jay + 1):
        i_m = m + max_jay
        m_phi_mat[i_m, :] = np.exp(-1j * m * phi_list) * np.diff(
            phi_list, append=phi_list[-1]
        )
    lam_j_m_theta_mat = np.zeros(
        (2, max_jay, 2 * max_jay + 1, len(theta_list)), dtype=complex
    )
    for i_lam in [0, 1]:
        lam = 1 if i_lam == 0 else -1
        for jay in range(1, max_jay + 1):
            i_jay = jay - 1
            fac_jay = np.sqrt((2 * jay + 1) / (4 * np.pi))
            for m in range(-jay, jay + 1):
                i_m = m + max_jay
                lam_j_m_theta_mat[i_lam, i_jay, i_m, :] = (
                    sp.wignersmalld(jay, m, lam, theta_list)
                    * np.diff(theta_list, append=theta_list[-1])
                    * np.sin(theta_list)
                    * fac_jay
                )
    vals_am = np.einsum(
        "ajme,mp,kaep->kajm", lam_j_m_theta_mat, m_phi_mat, vals_pw, optimize=True
    )
    return vals_am


