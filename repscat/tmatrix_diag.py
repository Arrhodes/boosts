"""
@author: Maxim Vavilin maxim@vavilin.de
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import copy
from wigners import wigner_3j

# from repscat.boost_tmat import boost_tmat_vals
import matplotlib as mpl
import repscat as rs


from repscat.boost_tmat import get_k1_k2_lists, boost_tmat_vals
from repscat.boost_tmat_at_k2 import get_k1_list, boost_tmat_at_k2_vals
from repscat.boost_tmat_at_k1 import get_k2_small_list #,boost_tmat_at_k1_vals_torch, boost_tmat_at_k1_vals_jax
from repscat.boost_tmat_at_k1 import boost_tmat_at_k1_vals, boost_tmat_at_k1_vals_torch

class TmatrixDiagonal:
    """Unitless T-matrix diagonal in frequency
    with arguments (k lam lam j j m m)
    and convention T(k1,k2) = T(k1) d(k1-k2)/k1
    """

    def __init__(self, k_list, vals, radius=None):
        self._check_domain_vs_vals(k_list.shape, vals.shape)
        self._k_list = k_list
        self._vals = vals
        self._radius = radius
        self._jay_max = vals.shape[4]

    def _check_domain_vs_vals(self, kshape, vshape):
        if not (kshape[0] == vshape[0]):
            raise ValueError(
                f"Domain shape does not correspond to values: {kshape[0]} to {vshape[0]}"
            )

    @property
    def k_list(self):
        return self._k_list

    @property
    def vals(self):
        return self._vals

    @property
    def radius(self):
        return self._radius

    @property
    def jay_max(self):
        return self._jay_max

    def scatter(self, incident):
        # if self._vals.shape[1::2] != incident.vals.shape:
        #     raise ValueError("Tmatrix shape does not correspond to incident k_list")
        if len(incident.k_list) == (len(self._k_list)) and np.allclose(
            self._k_list, incident.k_list
        ):
            scattered_vals = np.einsum(
                "kabijmn,kbjn->kaim", self._vals, incident.vals, optimize=True
            )
        else:
            if (
                incident.k_list[0] < self._k_list[0]
                or incident.k_list[-1] > self._k_list[-1]
            ):
                raise ValueError("Incident k_list outside of Tmat k_list")
            else:
                tmat_interpol = rs.interpolate_tmat(
                    incident.k_list, self._k_list, self._vals
                )
                scattered_vals = np.einsum(
                    "kabijmn,kbjn->kaim", tmat_interpol, incident.vals, optimize=True
                )
        return rs.WaveFunctionAngularMomentum(incident.k_list, scattered_vals)

    def plot(self):
        """Plots interaction cross section"""
        k_list = self._k_list
        interaction_cross_section = np.einsum(
            "kabijmn,kabijmn->k",
            np.conj(self._vals),  # (k,a,b,i,j,m,n)
            self._vals,  # (k,a,b,i,j,m,n)
            optimize=True,
        )
        plt.plot(k_list, np.real(interaction_cross_section))
        # plt.plot(k_list, np.real(self._vals[:,0,0,0,0,1,1]))
        # plt.plot(k_list, np.imag(self._vals[:,0,0,0,0,1,1]))
        # plt.xlabel("Wavenumber k, nm$^{-1}$")
        # plt.ylabel("Interaction cross section")

    def boost_precise(self, xi, k1_list, len_k2):
        k2_small_lists = np.zeros((len(k1_list), len_k2))
        boost_tmat_vals = np.zeros(
            (
                len(k1_list),
                len_k2,
                2,
                2,
                self._jay_max,
                self._jay_max,
                2 * self._jay_max + 1,
                2 * self._jay_max + 1,
            ),
            dtype=complex,
        )
        for i_k1, k1 in np.ndenumerate(k1_list):
            k2_small_lists[i_k1] = get_k2_small_list(xi, k1, len_k2)
            boost_tmat_vals[i_k1] = boost_tmat_at_k1_vals_torch(
                xi, self._k_list, self._vals, self._jay_max, k1, len_k2
            )

        return rs.TmatrixPolyPrecise(k1_list, k2_small_lists, boost_tmat_vals)

    def boost_tmat(self, xi, len_k1, len_k2):
        print("Boosting T-matrix")

        k1_list, k2_list = get_k1_k2_lists(self._k_list, xi, len_k1, len_k2)

        return rs.TmatrixPoly(
            k1_list,
            k2_list,
            boost_tmat_vals(
                xi, self._k_list, self._vals, self._jay_max, len_k1, len_k1
            ),
        )

    def boost_tmat_at_k2(self, xi, k2, len_k1):
        k1_list = get_k1_list(xi, k2, len_k1)

        return boost_tmat_at_k2_vals(
            xi, self._k_list, self._vals, self._jay_max, k2, len_k1
        )

    def boost_tmat_at_k1(self, xi, k1, len_k2): ## comment because already in precise method

        return boost_tmat_at_k1_vals(
            xi, self._k_list, self._vals, self._jay_max, k1, len_k2
        )

    def _get_indices(self, *indices):
        i_lam1 = 0 if indices[0] == 1 else 1
        i_lam2 = 0 if indices[1] == 1 else 1
        i_jay1 = indices[2] - 1
        i_jay2 = indices[3] - 1
        i_m1 = indices[4] + self._jay_max
        i_m2 = indices[5] + self._jay_max
        return i_lam1, i_lam2, i_jay1, i_jay2, i_m1, i_m2

    def compute_transfer(self, incident):
        scattered = self.scatter(incident)
        outgoing = incident + scattered

        return {
            "energy": incident.energy() - outgoing.energy(),
            "momentum_z": incident.momentum_z() - outgoing.momentum_z(),
        }
    
    def plot_norm(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        norms =  np.einsum(
            "kabijmn,kabijmn->k",
            np.conj(self._vals),  # (k,a,b,i,j,m,n)
            self._vals,  # (k,a,b,i,j,m,n)
            optimize=True
        )
        
        ax.plot(self._k_list*1000, np.sqrt(np.real(norms)), **kwargs) # MICROMETERS
        
        return ax

