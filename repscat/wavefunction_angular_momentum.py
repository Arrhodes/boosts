"""
@author: Maxim Vavilin maxim@vavilin.de
"""

import numpy as np
import matplotlib.pyplot as plt
from wigners import wigner_3j
import repscat as rs


class WaveFunctionAngularMomentum:
    def __init__(self, k_list, vals):
        if len(vals.shape) == 3:
            self._type = "monochromatic"
            self._k = k_list
        elif len(vals.shape) == 4:
            self._type = "polychromatic"
            self._k_list = k_list
        else:
            raise ValueError("Wrong shape of values")

        self._vals = vals
        self._jay_max = vals.shape[2]
        self._check_input()

    def _check_input(self):
        if self._type == "polychromatic":
            if self._vals.shape[0] != len(self._k_list):
                raise ValueError("Values do not correspond to k_list")

    def plot(self, linestyle="-", fac=1):
        if self._type == "polychromatic":
            for m in range(-self._jay_max, self._jay_max + 1):
                i_m = m + self._jay_max
                plt.plot(self._k_list, fac*self._vals[:, 0, :, i_m], linestyle=linestyle)
                plt.plot(self._k_list, fac*self._vals[:, 1, :, i_m], linestyle=linestyle)

    @property
    def k_list(self):
        return self._k_list

    @property
    def vals(self):
        return self._vals

    @property
    def jay_max(self):
        return self._jay_max

    def interpolate(self, k_list_new):
        if self._type == "polychromatic":
            interp_vals = np.zeros(
                (
                    len(k_list_new),
                    self._vals.shape[1],
                    self._vals.shape[2],
                    self._vals.shape[3],
                ),
                dtype=complex,
            )
            for i_lam in range(self._vals.shape[1]):
                for i_jay in range(self._vals.shape[2]):
                    jay = i_jay + 1
                    for m in range(-jay, jay + 1):
                        i_m = m + self._jay_max
                        real_part = np.interp(
                            k_list_new,
                            self._k_list,
                            np.real(self._vals[:, i_lam, i_jay, i_m]),
                        )
                        imag_part = np.interp(
                            k_list_new,
                            self._k_list,
                            np.imag(self._vals[:, i_lam, i_jay, i_m]),
                        )
                        interp_vals[:, i_lam, i_jay, i_m] = real_part + 1j * imag_part

            return WaveFunctionAngularMomentum(k_list_new, interp_vals)

    def __add__(self, other):
        if not (np.allclose(self._k_list, other.k_list)):
            raise ValueError("Domains of wavefunctions do not correspond")
        return WaveFunctionAngularMomentum(self._k_list, self._vals + other._vals)

    def photons(self):
        if self._type == "polychromatic":
            k_part = np.diff(self._k_list, append=self._k_list[-1]) * self._k_list
            photons = np.einsum(
                "kljm,k,kljm->", np.conj(self._vals), k_part, self._vals, optimize=True
            )
            return photons

    def helicity(self):
        if self._type == "polychromatic":
            k_part = np.diff(self._k_list, append=self._k_list[-1]) * self._k_list
            helicity_arr = np.array([1, -1])
            helicity = (
                np.einsum(
                    "l,kljm,k,kljm->",
                    helicity_arr,
                    np.conj(self._vals),
                    k_part,
                    self._vals,
                    optimize=True,
                )
                * rs.H_BAR_SI
            )
            return helicity

    def energy(self):
        if self._type == "polychromatic":
            dim_factor = rs.C_0_SI * rs.H_BAR_SI * 1e9  # Accounts for 1/nm
            k_part = np.diff(self._k_list, append=self._k_list[-1]) * self._k_list**2
            quantity = np.einsum(
                "kljm,k,kljm->", np.conj(self._vals), k_part, self._vals, optimize=True
            )
            return dim_factor * quantity

    def extend_values_by_one_jay(self):
        sh = self._vals.shape
        new_vals = np.zeros((sh[0], sh[1], sh[2] + 1, sh[3] + 2), dtype=complex)
        new_jay_max = sh[2] + 1
        for i_k in range(sh[0]):
            for i_lam in range(sh[1]):
                for i_jay in range(sh[2]):  # just old jay here
                    jay = i_jay + 1
                    for m in range(-jay, jay + 1):
                        i_m = m + new_jay_max  # crucial line
                        new_vals[i_k, i_lam, i_jay, i_m] = self._vals.shape[
                            i_k, i_lam, i_jay, i_m - 2
                        ]  # crusial line
        return new_jay_max, new_vals

    def momentum_z(self):
        if self._type == "polychromatic":
            # extended_j1_max = self._jay_max + 1
            dim_factor = rs.H_BAR_SI * 1e9  # Accounts for 1/nm in wavefunction
            k_part = np.diff(self._k_list, append=self._k_list[-1]) * self._k_list**2
            gamma_mat = np.zeros(
                (2, 2 * self._jay_max + 1, self._jay_max, self._jay_max), dtype=complex
            )
            for i_lam in [0, 1]:
                lam = 1 if i_lam == 0 else -1
                for jay1 in range(1, self._jay_max + 1):
                    i_jay1 = jay1 - 1
                    for m in range(-jay1, jay1 + 1):
                        i_m = m + self._jay_max
                        min_jay2 = max(1, jay1 - 1)
                        max_jay2 = min(
                            self.jay_max,
                            jay1 + 1,
                        )
                        for jay2 in range(min_jay2, max_jay2 + 1):
                            i_jay2 = jay2 - 1
                            if np.abs(m) < jay2 + 1:
                                gamma_mat[i_lam, i_m, i_jay1, i_jay2] = (
                                    np.sqrt(2 * jay1 + 1)
                                    * np.sqrt(2 * jay2 + 1)
                                    * (-1) ** (m - lam)
                                    * wigner_3j(jay1, jay2, 1, -m, m, 0)
                                    * wigner_3j(jay1, jay2, 1, -lam, lam, 0)
                                )
            quantity = np.einsum(
                "k,kljm,lmji,klim->",
                k_part,
                np.conj(self._vals),
                gamma_mat,
                self._vals,
                optimize=True,
            )
            return dim_factor * quantity

    def in_plane_wave_basis(self, theta_list, phi_list):
        pw_vals = rs.pw_from_am_vals(
            self._vals,
            theta_list,
            phi_list,
        )
        return rs.WaveFunctionPlaneWave(self._k_list, theta_list, phi_list, pw_vals)

    def boost(self, xi, max_jay_new=None):
        if max_jay_new is None:
            max_jay_new = self._jay_max
        new_k_list = rs.boost_wavenumbers(self.k_list, xi)
        new_vals = rs.boost_vals_am(
            self._vals, new_k_list, self._k_list, max_jay_new, xi
        )
        return rs.WaveFunctionAngularMomentum(new_k_list, new_vals)
