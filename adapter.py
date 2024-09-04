"""
@author: Maxim Vavilin maxim.vavilin@kit.edu
"""
import numpy as np
import repscat as rs
import matplotlib.pyplot as plt
import treams.special as sp


def get_boosted_coeffs_reference_am(k_list, xi, max_jay, k0, delta_k):
    """m=0, j=0"""
    k_list_boosted = rs.boost_k_list_uniformly(k_list, xi)
    jay = 1
    m = 0
    lam = 1
    fac = np.sqrt((2 * jay + 1))

    coeff_am_boosted = np.zeros((len(k_list_boosted), 2, max_jay, 2 * max_jay + 1))
    for i_k1, k1 in np.ndenumerate(k_list_boosted):
        # Integration domain
        k2_min = max(k_list[0], k1*np.exp(-xi))
        k2_max = min(k_list[-1], k1*np.exp(xi))
        k2_list = np.linspace(k2_min, k2_max, len(k_list))

        # Einsum parts
        k2_part = np.diff(k2_list, append=k2_list[-1])
        costheta1 = np.zeros_like(k2_list)
        costheta2 = np.zeros_like(k2_list)
        for i_k2, k2 in np.ndenumerate(k2_list):
            ct1 = (k1*np.cosh(xi)-k2)/(k1*np.sinh(xi))
            ct2 = (k1-k2*np.cosh(xi))/(k2*np.sinh(xi))
            if ct1 < -1:
                ct1 = -1
            if ct1 > 1:
                ct1 = 1
            if ct2 < -1:
                ct2 = -1
            if ct2 > 1:
                ct2 = 1
            costheta1[i_k2] = ct1
            costheta2[i_k2] = ct2
        
        wig1 = sp.wignersmalld(jay,m,lam,np.arccos(costheta1))
        wig2 = sp.wignersmalld(jay,m,lam,np.arccos(costheta2))
        gauss = np.exp(-((k2_list - k0) ** 2) / (2 * delta_k**2))
        
        coeff_am_boosted[i_k1, 0, 0, max_jay] = np.einsum(
            'k,k,k,k->',
            k2_part,
            wig1,
            wig2,
            gauss,
            optimize=True
        ) * 0.5 * fac**2 / (k1*np.sinh(xi))

    return k_list_boosted, coeff_am_boosted


def get_boosted_coeffs_reference(k_list, xi, max_jay, k0, delta_k):
    """m=0, j=0"""
    k_list_boosted = rs.boost_k_list_uniformly(k_list, xi)
    jay = 1
    m = 0
    lam = 1
    fac = np.sqrt((2 * jay + 1) / (4 * np.pi))
    eta_list = np.linspace(-1, 1, 400)
    coeff_pw_boosted = np.zeros((len(k_list_boosted), 400), dtype=complex)
    for i_k, k in np.ndenumerate(k_list_boosted):
        for i_eta, eta in np.ndenumerate(eta_list):
            eta_new = (eta * np.cosh(xi) + np.sinh(xi)) / (
                np.cosh(xi) + eta * np.sinh(xi)
            )
            k_new = k * np.cosh(xi) + k * eta * np.sinh(xi)
            coeff_pw_boosted[i_k, i_eta] = (
                fac
                * sp.wignersmalld(jay, m, lam, np.arccos(eta_new))
                * np.exp(-((k_new - k0) ** 2) / (2 * delta_k**2))
            )

    coeff_am_boosted = np.zeros((len(k_list_boosted), 2, max_jay, 2 * max_jay + 1), dtype = complex)
    eta_part = np.diff(eta_list, append=eta_list[-1]) * sp.wignersmalld(
        jay, m, lam, np.arccos(eta_list)
    )
    coeff_am_boosted[:, 0, 0, max_jay] = (
        np.einsum("e,ke->k", eta_part, coeff_pw_boosted, optimize=True) * 2 * np.pi * fac
    )

    return k_list_boosted, coeff_am_boosted


def get_boosted_coeffs_directly(coeffs, k_list, xi):
    ### Domain
    max_jay = np.shape(coeffs)[2]

    ### Create representation and boost
    rep_am = rs.Custom_pulse_wf_am(info_rep, domain, coeffs)

    boosted_am = rs.boost_am_representation(rep_am, max_jay, xi)

    return boosted_am.domain["k_list"], boosted_am.vals


def get_boosted_coeffs(coeffs, k_list, xi, num_theta=400, num_phi=150):
    if len(coeffs.shape)!= 4:
        raise ValueError('Coeffs of the wrong shape, must be (ks,lams,jays,ms)')
    ### Domain
    max_jay = np.shape(coeffs)[2]

    theta_list = np.linspace(0,np.pi,num_theta)
    phi_list = np.linspace(0,2*np.pi,num_phi)

    rep_am = rs.WaveFunctionAngularMomentum(k_list, coeffs)
    rep_pw = rep_am.in_plane_wave_basis(theta_list, phi_list)
    
    boosted_pw = rep_pw.boost(xi)
    
    boosted_am = boosted_pw.in_angular_momentum_basis(max_jay)
    
    return boosted_am.k_list, boosted_am.vals
