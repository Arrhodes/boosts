"""
@author: Maxim Vavilin maxim@vavilin.de
"""
import numpy as np
from reptemscat.field import Representation, Adapter
from reptemscat.aux_funcs import (
    get_k_list_gaussian,
    gaussian_wavenumber,
    get_k_list_narrow_gaussian,
)
import treams.special as sp
from reptemscat.constants import C_0
from reptemscat.aux_funcs import cartesian_product

# import itertools
from numpy.linalg import multi_dot


class WaveFunctionAngularMomentum(Representation):
    def _get_domain(self):
        return {"k_list": Adapter.get_k_list_gaussian(self._info)}

    def _compute_vals(self):
        k_0 = 2 * np.pi / self._info.parameters["center_wavelength"]
        dt = self._info.parameters["width_time"]
        vals = np.zeros((self._info.parameters["num_k"]), dtype=complex)
        for i_k, k in np.ndenumerate(self._domain["k_list"]):
            vals[i_k] = gaussian_wavenumber(k, k_0, dt)
        return vals

    def _check_info(self, info):
        if info.field_name != "SphericalPulse":
            raise ValueError("Wrong physical field")
        if info.representation_name != "WF_AM":
            raise ValueError("Wrong wave function type")
        if not {"center_wavelength", "width_time", "num_k"} <= set(info.parameters):
            raise ValueError("Need more info")


class WaveFunctionPlaneWave(Representation):
    def _get_domain(self):
        return {
            "k_list": get_k_list_narrow_gaussian(
                self.info.parameters["center_wavelength"],
                self.info.parameters["width_time"],
                self.info.parameters["num_k"],
            ),
            "eta_list": np.linspace(
                -1, 1, self._info.parameters["num_eta"], endpoint=True
            ),
            "phi_list": np.linspace(
                0, 2 * np.pi, self._info.parameters["num_phi"], endpoint=True
            ),
        }

    def _compute_vals(self):
        J = self.info.parameters["J"]
        m = self.info.parameters["m"]
        lam = self.info.parameters["lam"]
        dt = self.info.parameters["width_time"]
        sqrt_fac = np.sqrt((2 * J + 1) / (4 * np.pi))

        k_0 = 2 * np.pi / self.info.parameters["center_wavelength"]
        k_list = self.domain["k_list"]
        eta_list = self.domain["eta_list"]
        phi_list = self.domain["phi_list"]
        Dt2C2 = dt**2 * C_0**2 / 2

        small_d_list = sp.wignersmalld(J, m, lam, np.arccos(eta_list))
        exp_k_list = np.exp(-((k_list - k_0) ** 2) * Dt2C2)
        exp_phi_list = np.exp(1j * phi_list * m)

        A, B, C = np.ix_(exp_k_list, small_d_list, exp_phi_list)

        vals = np.zeros(
            (
                2,
                self.info.parameters["num_k"],
                self.info.parameters["num_eta"],
                self.info.parameters["num_phi"],
            ),
            dtype=complex,
        )
        i_lam = 0 if lam == 1 else 1
        vals[i_lam, :, :, :] = sqrt_fac * A * B * C

        return vals


    def _check_info(self, info):
        if info.field_name != "SphericalPulse":
            raise ValueError("Wrong physical field")
        if info.representation_name != "WF_PW":
            raise ValueError("Wrong wave function type")
        if not {
            "num_k",
            "num_eta",
            "num_phi",
            "center_wavelength",
            "width_time",
            "norm_fac",
        } <= set(info.parameters):
            raise ValueError("Need more info")
